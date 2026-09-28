
class AttributeBooster:
    """
    Boosts candidates based on query attributes and entity overlap.

    Two-stage boosting:
    1. Entity overlap: Boosts based on shared entities between query and memory
    2. Attribute mapping: Boosts based on attribute_map configuration

    Entity and attribute boosts are stored separately in diagnostics.
    """

    def __init__(
        self,
        attribute_map=None,
        boost_value=0.15,
        entity_boost=0.10,
    ):
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

        # `detected` is keyed by field, while `alias_index` contains
        # one entry per canonical attribute and alias. Cache the unique
        # field count so the detection fast path compares like with like.
        self._attribute_fields = {
            meta["field"]
            for meta in self.alias_index.values()
        }

    def _build_alias_index(self):
        """Build alias index for fast attribute lookups."""
        index = {}

        for canonical, config in self.attribute_map.items():
            field = config.get("field", canonical)
            boost = config.get("boost", self.boost_value)
            aliases = config.get("aliases", [])

            canonical_lower = canonical.lower()

            index[canonical_lower] = {
                "field": field,
                "boost": boost,
            }

            for alias in aliases:
                alias_lower = alias.lower()

                index[alias_lower] = {
                    "field": field,
                    "boost": boost,
                }

        return index

    def _detect_attributes(self, query):
        """
        Detect attributes from query text and tokens.

        Uses token matching for performance, with text fallback
        for multi-word attributes.

        The fast path compares detected unique fields against
        the number of unique configured fields. This avoids
        incorrectly comparing field count against the larger
        canonical/alias entry count.
        """
        detected = {}

        text = query.normalized_text.lower()
        tokens = [
            token.lower()
            for token in query.tokens
        ]

        # Fast token-based detection.
        for token in tokens:
            hit = self.alias_index.get(token)

            if hit:
                field = hit["field"]

                detected[field] = max(
                    detected.get(field, 0.0),
                    hit["boost"],
                )

        # Only scan aliases in the text when token matching did
        # not already detect every configured attribute field.
        #
        # This is especially useful for multi-word attributes that
        # cannot be represented as a single query token.
        if len(detected) < len(self._attribute_fields):
            for alias, meta in self.alias_index.items():
                if alias in text and alias not in tokens:
                    field = meta["field"]

                    detected[field] = max(
                        detected.get(field, 0.0),
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
            entity.lower()
            if isinstance(entity, str)
            else str(entity).lower()
            for entity in query_entities
        }

        memory_set = {
            entity.lower()
            if isinstance(entity, str)
            else str(entity).lower()
            for entity in memory_entities
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
        """
        Check if an attribute matches metadata.

        `boost_value` is retained in the signature for compatibility
        with existing callers, although matching itself does not depend
        on its value.
        """
        if not metadata:
            return False

        if attr in metadata:
            return True

        attr_lower = attr.lower()

        for key, value in metadata.items():
            if (
                isinstance(value, str)
                and attr_lower in value.lower()
            ):
                return True

            if (
                isinstance(key, str)
                and attr_lower in key.lower()
            ):
                return True

        return False

    def boost(self, query, candidates):
        """
        Apply attribute and entity-based boosting to candidates.

        Stores entity and attribute contributions separately in
        candidate diagnostics and dedicated ranking fields.
        """
        if not candidates:
            return candidates

        query_entities = query.entities
        detected_attributes = self._detect_attributes(query)

        for candidate in candidates:
            entity_score = 0.0
            attribute_score = 0.0
            overlap_entities = []

            # --------------------------------------------------------
            # Entity overlap boost
            # --------------------------------------------------------

            if query_entities:
                entity_score, overlap = self._entity_overlap_score(
                    query_entities,
                    candidate.memory.entities,
                )

                overlap_entities = overlap

            # --------------------------------------------------------
            # Attribute boost
            # --------------------------------------------------------

            if detected_attributes:
                memory_type = candidate.memory.memory_type
                metadata = candidate.memory.metadata or {}

                if (
                    memory_type
                    and memory_type in detected_attributes
                ):
                    attribute_score += detected_attributes[
                        memory_type
                    ]

                for attr, boost_value in detected_attributes.items():
                    if self._matches_metadata(
                        metadata,
                        attr,
                        boost_value,
                    ):
                        attribute_score += boost_value * 0.5

            # --------------------------------------------------------
            # Apply computed boost
            # --------------------------------------------------------

            candidate.entity_score = entity_score
            candidate.attribute_score = attribute_score

            total_boost = (
                entity_score
                + attribute_score
            )

            # --------------------------------------------------------
            # Diagnostics
            # --------------------------------------------------------

            candidate.diagnostics["entity_boost"] = entity_score
            candidate.diagnostics["attribute_boost"] = attribute_score
            candidate.diagnostics["total_boost"] = total_boost
            candidate.diagnostics["attribute_overlap"] = overlap_entities
            candidate.diagnostics["detected_attributes"] = list(
                detected_attributes.keys()
            )

        return candidates
