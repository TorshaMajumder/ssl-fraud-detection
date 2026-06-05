

-- 1. Enable the pgvector extension to work with embeddings
create extension if not exists vector;

-- 2. Create the Transactions table (The "Event Log")
create table transactions (
  id uuid primary key default uuid_generate_v4(),
  user_id text,
  amount float,
  merchant text,
  location text,
  embedding vector(384),
  is_anomaly boolean,
  human_feedback boolean default null, -- true = True Positive, false = False Positive
  created_at timestamp with time zone default now()
);

-- 3. Create the User Profiles table (The "Behavioral Anchors")
create table user_profiles (
  user_id text primary key,
  full_name text,
  account_tier text, -- e.g., 'gold', 'silver'
  behavioral_embedding vector(384), -- This is the average 'normal' SSL vector for this user
  updated_at timestamp with time zone default now()
);

-- 4. Create the System Configuration table (The "Dynamic Threshold")
create table system_config (
  key text primary key,
  value float
);

-- 5. Seed the initial threshold for the anomaly detection logic
insert into system_config (key, value) values ('anomaly_threshold', 0.85);


-- 6. Enable Row Level Security (RLS) for Production Best Practices
alter table transactions enable row level security;
alter table user_profiles enable row level security;
alter table system_config enable row level security;


update system_config set value = 18.0 where key = 'anomaly_threshold';

-- Note: In this project, we access data via the Service Role Key from the backend, 
-- which bypasses RLS. For public-facing apps, specific RLS Policies would be added here.