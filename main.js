import { ChatOpenAI } from "@langchain/openai";
import { loadSummarizationChain } from "langchain/chains";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "langchain/document";
import { PromptTemplate } from "@langchain/core/prompts";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from 'url';
import 'dotenv/config';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function summarizeTranscript() {
  try {
    console.log("Starting transcript summarization...");

    // Read the transcript file
    const transcriptPath = path.join(__dirname, "transcripts", "my-first-million-how-to-master-storytelling", "openai-whisper.txt");
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
      modelName: "gpt-4o-2024-08-06",
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

    // Create the summarization chain
    const chain = loadSummarizationChain(model, {
      type: "map_reduce",
      // verbose: true,
      mapPrompt: mapPrompt,
      combinePrompt: combinePrompt,
    });

    // Run the summarization chain
    console.log("Generating summary...");
    const result = await chain.call({
      input_documents: docs,
    });

    // Save the result
    console.log("Saving the summary...");
    const outputPath = path.join(__dirname, "output", "summary.md");
    await fs.writeFile(outputPath, result.text);
    console.log(`Summary saved to ${outputPath}`);

    console.log("Transcript summarization completed successfully.");
  } catch (error) {
    console.error("Error summarizing transcript:", error);
  }
}

summarizeTranscript();