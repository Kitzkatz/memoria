class AttributeBooster:
    """
    Boosts candidates based on query attributes and entity overlap.

    Two-stage boosting:
    1. Entity overlap: Boosts based on shared entities between query and memory
    2. Attribute mapping: Boosts based on attribute_map configuration

    Entity and attribute boosts are stored separately in diagnostics.
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

        # For faster text scanning, cache lowercased aliases
        self._aliases_lower = [
            a.lower()
            for a in self.alias_index.keys()
        ]

    def _build_alias_index(self):
        """Build alias index for fast attribute lookups."""
        index = {}

        for canonical, config in self.attribute_map.items():
            field = config.get("field", canonical)
            boost = config.get("boost", self.boost_value)
            aliases = config.get("aliases", [])

            index[canonical.lower()] = {
                "field": field,
                "boost": boost,
            }

            for alias in aliases:
                index[alias.lower()] = {
                    "field": field,
                    "boost": boost,
                }

        return index

    def _detect_attributes(self, query):
        """
        Detect attributes from query text and tokens.

        Uses token matching for performance, with text fallback
        for multi-word attributes.
        """
        detected = {}

        text = query.normalized_text.lower()
        tokens = [t.lower() for t in query.tokens]

        for token in tokens:
            hit = self.alias_index.get(token)

            if hit:
                detected[hit["field"]] = max(
                    detected.get(hit["field"], 0.0),
                    hit["boost"],
                )

        if len(detected) < len(self.alias_index):
            for alias, meta in self.alias_index.items():
                if alias in text and alias not in tokens:
                    detected[meta["field"]] = max(
                        detected.get(meta["field"], 0.0),
                        meta["boost"],
                    )

        return detected

    def _entity_overlap_score(
        self,
        query_entities,
        memory_entities,
    ):
        """Calculate entity overlap score."""
        if not query_entities or not memory_entities:
            return 0.0, []

        query_set = {
            e.lower()
            if isinstance(e, str)
            else str(e).lower()
            for e in query_entities
        }

        memory_set = {
            e.lower()
            if isinstance(e, str)
            else str(e).lower()
            for e in memory_entities
        }

        overlap = query_set & memory_set

        return (
            self.entity_boost * len(overlap),
            list(overlap),
        )

    def _matches_metadata(
        self,
        metadata,
        attr,
        boost_value,
    ):
        """Check if an attribute matches metadata."""
        if not metadata:
            return False

        if attr in metadata:
            return True

        for key, value in metadata.items():
            if (
                isinstance(value, str)
                and attr in value.lower()
            ):
                return True

            if (
                isinstance(key, str)
                and attr in key.lower()
            ):
                return True

        return False

    def boost(self, query, candidates):
        """
        Apply attribute and entity-based boosting to candidates.

        Stores entity and attribute contributions separately in
        candidate diagnostics.
        """
        if not candidates:
            return candidates

        query_entities = query.entities
        detected_attributes = self._detect_attributes(query)

        for candidate in candidates:
            entity_score = 0.0
            attribute_score = 0.0
            overlap_entities = []

            # Entity overlap boost
            if query_entities:
                entity_score, overlap = self._entity_overlap_score(
                    query_entities,
                    candidate.memory.entities,
                )

                overlap_entities = overlap

            # Attribute boost
            if detected_attributes:
                memory_type = candidate.memory.memory_type
                metadata = candidate.memory.metadata or {}

                if (
                    memory_type
                    and memory_type in detected_attributes
                ):
                    attribute_score += detected_attributes[memory_type]

                for attr, boost_value in detected_attributes.items():
                    if self._matches_metadata(
                        metadata,
                        attr,
                        boost_value,
                    ):
                        attribute_score += boost_value * 0.5

            total_boost = entity_score + attribute_score

            candidate.diagnostics["entity_boost"] = entity_score
            candidate.diagnostics["attribute_boost"] = attribute_score
            candidate.diagnostics["total_boost"] = total_boost
            candidate.diagnostics["attribute_overlap"] = overlap_entities
            candidate.diagnostics["detected_attributes"] = list(
                detected_attributes.keys()
            )

        return candidates
