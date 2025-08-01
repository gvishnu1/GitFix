-- Vector similarity functions for PostgreSQL
-- This file contains functions for vector operations in PostgreSQL

-- Function to calculate dot product between two vectors
-- The first argument is a numeric array (vector)
-- The second argument is a string representation of a vector that needs to be converted
CREATE OR REPLACE FUNCTION dot_product(vector1 NUMERIC[], vector2 VARCHAR)
RETURNS FLOAT8 AS $$
DECLARE
    vector2_array NUMERIC[];
BEGIN
    -- Convert the string representation to an array
    -- The string format is expected to be "[0.1, 0.2, 0.3, ...]"
    vector2_array := string_to_array(
                        trim(both '[]' from vector2),
                        ','
                     )::NUMERIC[];

    -- Calculate dot product
    RETURN (
        SELECT SUM(v1 * v2)
        FROM unnest(vector1, vector2_array) AS t(v1, v2)
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to calculate cosine similarity between two vectors
CREATE OR REPLACE FUNCTION cosine_similarity(vector1 NUMERIC[], vector2 VARCHAR)
RETURNS FLOAT8 AS $$
DECLARE
    dot FLOAT8;
    norm1 FLOAT8;
    norm2 FLOAT8;
    vector2_array NUMERIC[];
BEGIN
    -- Convert the string representation to an array
    vector2_array := string_to_array(
                        trim(both '[]' from vector2),
                        ','
                     )::NUMERIC[];

    -- Calculate dot product
    dot := (
        SELECT SUM(v1 * v2)
        FROM unnest(vector1, vector2_array) AS t(v1, v2)
    );

    -- Calculate magnitudes
    norm1 := (
        SELECT SQRT(SUM(v * v))
        FROM unnest(vector1) AS v
    );

    norm2 := (
        SELECT SQRT(SUM(v * v))
        FROM unnest(vector2_array) AS v
    );

    -- Return cosine similarity
    IF norm1 = 0 OR norm2 = 0 THEN
        RETURN 0;
    ELSE
        RETURN dot / (norm1 * norm2);
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;