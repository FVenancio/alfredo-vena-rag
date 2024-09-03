import { openai } from "@ai-sdk/openai";
import { convertToCoreMessages, streamText, tool } from "ai";
import { z } from "zod";
import { findRelevantContent } from "@/lib/ai/embedding";

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages } = await req.json();

  const initial_prompt = `
You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information. 
After you have rewritten the query, check your knowledge base before answering any questions. 
Only respond to questions using information from tool calls. 
If no relevant information is found in the tool calls, apology and answer that you don't have that information available. 
Always answer back using the same language the user spoke in. If the user used Portuguese, answer back using Portuguese from Portugal.`;

  const result = await streamText({
    model: openai("gpt-4o-mini"),
    system: initial_prompt,
    messages: convertToCoreMessages(messages),
    tools: {
      getInformation: tool({
        description: `get information from your knowledge base to answer questions.`,
        parameters: z.object({
          question: z.string().describe("the users question"),
        }),
        execute: async ({ question }) => findRelevantContent(question),
      }),
    },
  });

  return result.toDataStreamResponse();
}
