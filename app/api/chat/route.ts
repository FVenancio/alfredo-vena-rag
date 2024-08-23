import { openai } from "@ai-sdk/openai";
import { convertToCoreMessages, streamText, tool } from "ai";
import { z } from "zod";
import { findRelevantContent } from "@/lib/ai/embedding";

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = await streamText({
    model: openai("gpt-4o-mini"),
    system: `You are a helpful assistant. Check your knowledge base before answering any questions.
    Only respond to questions using information from tool calls.
    if no relevant information is found in the tool calls, respond, "Sorry, I don't know."`,
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
