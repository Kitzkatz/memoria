class AttributeBooster:
    """
    Boosts candidates based on query attributes and entity overlap.

    Two-stage boosting:
    1. Entity overlap: Boosts based on shared entities between query and memory
    2. Attribute mapping: Boosts based on attribute_map configuration (if provided)

    The final score is the sum of both signals.
    """

    def __init__(self, attribute_map=None, boost_value=0.15, entity_boost=0.10):
        """
        Args:
            attribute_map: Optional dict mapping attributes to boost values
            boost_value: Default boost for attribute matches
            entity_boost: Boost per matching entity
        """
        self.boost_value = boost_value
        self.entity_boost = entity_boost
        self.attribute_map = attribute_map or {}
        self.alias_index = self._build_alias_index()

    def _build_alias_index(self):
        """Build alias index for fast attribute lookups."""
        index = {}
        for canonical, config in self.attribute_map.items():
            field = config.get("field", canonical)
            boost = config.get("boost", self.boost_value)
            aliases = config.get("aliases", [])

            index[canonical.lower()] = {"field": field, "boost": boost}
            for alias in aliases:
                index[alias.lower()] = {"field": field, "boost": boost}

        return index

    def _detect_attributes(self, query):
        """Detect attributes from query text and tokens."""
        detected = {}
        text = query.normalized_text.lower()
        tokens = [t.lower() for t in query.tokens]

        # Check tokens against alias index
        for token in tokens:
            hit = self.alias_index.get(token)
            if hit:
                detected[hit["field"]] = max(
                    detected.get(hit["field"], 0.0),
                    hit["boost"]
                )

        # Check full text for multi-word attributes
        for alias, meta in self.alias_index.items():
            if alias in text:
                detected[meta["field"]] = max(
                    detected.get(meta["field"], 0.0),
                    meta["boost"]
                )

        return detected

    def _entity_overlap_score(self, query_entities, memory_entities):
        """Calculate entity overlap score."""
        if not query_entities or not memory_entities:
            return 0.0

        query_set = {e.lower() for e in query_entities}
        memory_set = {e.lower() for e in memory_entities}

        overlap = query_set & memory_set
        return self.entity_boost * len(overlap), list(overlap)

    def boost(self, query, candidates):
        """
        Apply attribute and entity-based boosting to candidates.

        Args:
            query: QueryRecord object
            candidates: List of CandidateRecord objects

        Returns:
            List of CandidateRecord objects with attribute_score set
        """
        if not candidates:
            return candidates

        query_entities = query.entities
        detected_attributes = self._detect_attributes(query)

        for candidate in candidates:
            total_boost = 0.0
            overlap_entities = []

            # 1. Entity overlap boost
            if query_entities:
                entity_score, overlap = self._entity_overlap_score(
                    query_entities,
                    candidate.memory.entities
                )
                total_boost += entity_score
                overlap_entities = overlap

            # 2. Attribute boost
            if detected_attributes:
                # Check if candidate matches any detected attribute
                # Could check memory_type, metadata, or custom fields
                memory_type = candidate.memory.memory_type
                if memory_type in detected_attributes:
                    total_boost += detected_attributes[memory_type]

                # Also check metadata for attribute matches
                metadata = candidate.memory.metadata or {}
                for attr, boost_value in detected_attributes.items():
                    if attr in metadata or attr in str(metadata.values()):
                        total_boost += boost_value * 0.5

            # Set the score
            candidate.attribute_score = total_boost
            candidate.diagnostics["attribute_boost"] = total_boost
            candidate.diagnostics["attribute_overlap"] = overlap_entities
            candidate.diagnostics["detected_attributes"] = list(detected_attributes.keys())

        return candidates
