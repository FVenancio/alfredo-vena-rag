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
Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information from documents.
After you have rewritten the query, check your knowledge base before answering any questions. 
Only respond to questions using information from tool calls.
The available documents are relevant to the real estate market and are always one of these types:
  - "certidao_registo_predial": "Document to prove who the owner of the property is and if are any outstanding debts"
  - "caderneta_predial": "Document that shows all details regarting the property"
  - "licenca_utilizacao": "Document that proves that the property is legally allowed for its indended purpose"
  - "certidao_isencao": ""
  - "certidao_infraestruturas": "Document that proves that the construction of the property has finished"
  - "ficha_tecnica_habitacao": "Document showcasing the technical and functional characteristics of an urban building for housing purposes, reported at the time of completion of construction, reconstruction, expansion or alteration works.
  - "certificado_energetico": "Document that evaluates the energy efficiency of a property on a scale from A+ (very efficient) to F (not very efficient)"
  - "planta_imovel": "Property plan"
  - "documento_kyc": ""
  - "documento_preferencia": ""

Note: the "certificado_energetico" (enery certificate) document follows a very specific structure in which the energy class given is presented in image format in the document's header.

If no relevant information is found in the tool calls, apology and answer that you don't have that information available. 
Always answer back using the same language the user spoke in. If the user used Portuguese, answer back using Portuguese from Portugal.`;

  const result = await streamText({
    model: openai("gpt-4o-mini"),
    system: initial_prompt,
    messages: convertToCoreMessages(messages),
    tools: {
      getInformation: tool({
        description: `Get information from your documents knowledge base to answer questions.`,
        parameters: z.object({
          question: z.string().describe("the users question"),
          document_type: z
            .enum([
              "certidao_registo_predial",
              "caderneta_predial",
              "licenca_utilizacao",
              "certidao_isencao",
              "certidao_infraestruturas",
              "ficha_tecnica_habitacao",
              "certificado_energetico",
              "planta_imovel",
              "documento_kyc",
              "documento_preferencia",
              "null",
            ])
            .default("null")
            .describe(
              "The specific type of document the user wants to get information from. If you're not sure which, use 'null'."
            ),
        }),
        execute: async ({ question, document_type }) =>
          findRelevantContent(question, document_type),
      }),
    },
  });

  return result.toDataStreamResponse();
}
