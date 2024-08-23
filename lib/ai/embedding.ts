import { env } from "@/lib/env.mjs";

import { embed, embedMany } from "ai";
import { openai } from "@ai-sdk/openai";
import { createClient } from "@supabase/supabase-js";

const embeddingModel = openai.embedding("text-embedding-3-small");
const supabase = createClient(env.SUPABASE_URL, env.SUPABASE_KEY);

const generateChunks = (input: string): string[] => {
  return input
    .trim()
    .split(".")
    .filter((i) => i !== "");
};

export const generateEmbeddings = async (
  value: string
): Promise<Array<{ embedding: number[]; content: string }>> => {
  const chunks = generateChunks(value);
  const { embeddings } = await embedMany({
    model: embeddingModel,
    values: chunks,
  });
  return embeddings.map((e, i) => ({ content: chunks[i], embedding: e }));
};

export const generateEmbedding = async (value: string): Promise<number[]> => {
  const input = value.replaceAll("\\n", " ");
  const { embedding } = await embed({
    model: embeddingModel,
    value: input,
  });
  return embedding;
};

export const findRelevantContent = async (userQuery: string) => {
  const userQueryEmbedded = await generateEmbedding(userQuery);
  const { data: documents } = await supabase.rpc("match_documents", {
    query_embedding: userQueryEmbedded,
    match_threshold: 0.5, // Data threshold
    match_count: 4, // Number of matches
  });

  return JSON.stringify(documents);
};
