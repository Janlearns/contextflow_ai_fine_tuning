-- Jalankan query ini di Supabase SQL Editor

CREATE TABLE IF NOT EXISTS users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT CHECK (role IN ('user', 'assistant')) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS predictions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id TEXT NOT NULL,
    instruction TEXT NOT NULL,
    context TEXT,
    response TEXT NOT NULL,
    model_version TEXT,
    latency_ms FLOAT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    prediction_id UUID REFERENCES predictions(id) ON DELETE CASCADE,
    rating INT CHECK (rating BETWEEN 1 AND 5),
    comment TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS system_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    level TEXT,
    message TEXT,
    module TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_metadata (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    model_name TEXT NOT NULL,
    base_model TEXT,
    bleu_score FLOAT,
    rouge1 FLOAT,
    rougeL FLOAT,
    trained_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS training_datasets (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    model_id UUID REFERENCES model_metadata(id) ON DELETE CASCADE,
    instruction TEXT NOT NULL,
    context TEXT,
    response TEXT NOT NULL,
    source TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS company_data (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    city TEXT,
    company_name TEXT NOT NULL,
    instruction TEXT NOT NULL,
    context TEXT,
    response TEXT NOT NULL,
    source_file TEXT,
    source_type TEXT CHECK (source_type IN ('csv', 'pdf', 'image', 'json', 'docx', 'txt')),
    created_at TIMESTAMPTZ DEFAULT now()
);