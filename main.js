import { ChatOpenAI } from "@langchain/openai";
import { loadSummarizationChain } from "langchain/chains";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "langchain/document";
import { PromptTemplate } from "@langchain/core/prompts";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from 'url';
import 'dotenv/config';
import { z } from "zod";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function summarizeTranscript() {
  try {
    console.log("Starting transcript summarization...");

    // Read the transcript file
    const transcriptPath = path.join(__dirname, "transcripts", "my-first-million-0-to-1-million", "openai-whisper.txt");
    const transcript = await fs.readFile(transcriptPath, "utf-8");
    console.log("Transcript loaded.");

    // Initialize the text splitter
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 10000,
      chunkOverlap: 200,
    });

    // Split the transcript
    console.log("Splitting transcript into chunks...");
    const docs = await textSplitter.splitDocuments([
      new Document({ pageContent: transcript }),
    ]);
    console.log(`Transcript split into ${docs.length} chunks.`);

    // Initialize the OpenAI model
    console.log("Initializing OpenAI model...");
    const model = new ChatOpenAI({
      temperature: 0.3,
      modelName: "gpt-4o-mini-2024-07-18",
      maxTokens: 4000,
    });

    // Custom prompt for individual chunks
    const mapPrompt = PromptTemplate.fromTemplate(`
Analyze the following part of a video transcript and create a partial summary. Focus on the content of this specific part.

# Summary Sections:
- Key Points: List the main topics or concepts discussed in this part as bullet points.
- Detailed Notes: Provide a structured summary of this part's content. Include key arguments, examples, and explanations.
- Quotes: Extract any notable quotes or important statements from this part.
- Actions/Takeaways: List any practical tips or lessons mentioned in this part.
- Terms: Define any specialized terms or concepts introduced in this part.

# Transcript Part:
{text}

Provide a concise summary focusing on the above sections:
`);

    // Custom prompt for the final summary
    const combinePrompt = PromptTemplate.fromTemplate(`
Create a comprehensive cheatsheet for the entire video content by synthesizing the following partial summaries. Organize the information logically and eliminate redundancies.

{text}

Generate a final cheatsheet with these sections:
- Summary: Brief overview capturing the essence of the entire video content.
- Key Points/Main Ideas: Consolidated list of main topics or concepts discussed.
- Detailed Notes: Comprehensive and structured summary of the video content, including key arguments, examples, and explanations. Use headings and bullet points.
- Important Quotes: List of the most notable quotes or standout statements.
- Actions/Takeaways: Compiled list of practical tips, steps, or lessons viewers can apply.
- Glossary: Definitions of important specialized terms or concepts introduced.
- References and Resources: Any external resources or citations mentioned.

Ensure the final cheatsheet is well-organized and covers the entire video content:
`);

    const outputSchema = z.object({
      summary: z.string().describe("Brief overview capturing the essence of the entire video content."),
      keyPoints: z.array(z.string()).describe("Consolidated list of main topics or concepts discussed."),
      detailedNotes: z.array(z.string()).describe("Comprehensive and structured summary of the video content, including key arguments, examples, and explanations. Use headings and bullet points."),
      importantQuotes: z.array(z.string()).describe("List of the most notable quotes or standout statements."),
      actionsTakeaways: z.array(z.string()).describe("Compiled list of practical tips, steps, or lessons viewers can apply."),
      glossary: z.array(z.string()).describe("Definitions of important specialized terms or concepts introduced."),
      referencesAndResources: z.array(z.string()).describe("Any external resources or citations mentioned."),
    });

    // Create the summarization chain
    const chain = loadSummarizationChain(model, {
      type: "map_reduce",
      // verbose: true,
      mapPrompt: mapPrompt,
      combinePrompt: combinePrompt,
    });

    // Run the summarization chain
    console.log("Generating summary...");
    const result = await chain.invoke({
      input_documents: docs,
    });

    // Call again the model with structured output -> TODO: find a way to do that without calling it again
    console.log("Generating structured output...");
    const structuredLLM = model.withStructuredOutput(outputSchema, {
      strict: true,
    });
    const structuredOutputPrompt = PromptTemplate.fromTemplate(`
      Generate a JSON output based on the following text:
      {text}
    `);
    const structuredOutput = await structuredLLM.invoke(
      await structuredOutputPrompt.formatPromptValue({ text: result.text })
    );
    console.log(structuredOutput);

    // Save the result as JSON and text
    console.log("Saving the summary as JSON and text...");
    const jsonOutputPath = path.join(__dirname, "output", "summary.json");
    await fs.writeFile(jsonOutputPath, JSON.stringify(structuredOutput, null, 2));
    console.log(`JSON summary saved to ${jsonOutputPath}`);

    const textOutputPath = path.join(__dirname, "output", "summary.md");
    await fs.writeFile(textOutputPath, result.text);
    console.log(`Text summary saved to ${textOutputPath}`);

    console.log("Transcript summarization completed successfully.");
  } catch (error) {
    console.error("Error summarizing transcript:", error);
  }
}

summarizeTranscript();