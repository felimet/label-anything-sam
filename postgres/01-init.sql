-- Label Studio PostgreSQL init script
-- Runs once on first container creation.

-- Extensions for full-text search and annotation data
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Label Studio creates its own tables via Django migrations on startup.
-- No manual table creation needed here.
