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

// Define output schema
const outputSchema = z.object({
  summary: z.string().describe("Brief overview capturing the essence of the entire video content."),
  keyPoints: z.array(z.string()).describe("Consolidated list of main topics or concepts discussed."),
  detailedNotes: z.array(z.string()).describe("Comprehensive and structured summary of the video content, including key arguments, examples, and explanations."),
  importantQuotes: z.array(z.string()).describe("List of the most notable quotes or standout statements."),
  actionsTakeaways: z.array(z.string()).describe("Compiled list of practical tips, steps, or lessons viewers can apply."),
  glossary: z.array(z.string()).describe("Definitions of important specialized terms or concepts introduced."),
  referencesAndResources: z.array(z.string()).describe("Any external resources or citations mentioned."),
});

// Define theme schema
const themeSchema = z.object({
  themes: z.array(z.object({
    title: z.string().describe("A short title summarizing the theme."),
    description: z.string().describe("A brief summary explaining what is said about this theme."),
    subThemes: z.array(z.string()).describe("List of sub-topics related to the main theme. Can be an empty array if there are no sub-themes."),
  })),
});

// Part 1: Transcript Processing
async function processTranscript(transcriptPath) {
  console.log("Starting transcript processing...");
  
  console.log("Loading and splitting transcript...");
  const transcript = await fs.readFile(transcriptPath, "utf-8");
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 10000,
    chunkOverlap: 200,
  });
  const docs = await textSplitter.splitDocuments([
    new Document({ pageContent: transcript }),
  ]);
  console.log(`Transcript split into ${docs.length} chunks.`);
  
  return docs;
}

// New Part 2: Theme Identification
async function identifyThemes(docs) {
  console.log("Starting theme identification...");

  const model = initializeOpenAIModel();
  const themeChain = loadSummarizationChain(model, {
    type: "map_reduce",
    mapPrompt: PromptTemplate.fromTemplate(`
Analyze the following part of a video transcript and identify the themes or topics discussed. For each theme, provide:

- **Theme Title**: A short title summarizing the theme.
- **Description**: A brief summary explaining what is said about this theme.
- **Sub-themes** (if applicable): List of sub-topics related to the main theme.

Transcript part:
{text}

Identify the themes in this part:
    `),
    combinePrompt: PromptTemplate.fromTemplate(`
Combine and consolidate the following theme identifications from different parts of the transcript. Eliminate redundancies and organize the themes logically.

{text}

Provide a final list of main themes for the entire transcript:
    `),
  });

  console.log("Identifying themes...");
  const result = await themeChain.invoke({ input_documents: docs });

  console.log("Generating structured theme output...");
  const structuredThemes = await generateStructuredThemes(model, result.text);

  // Call the new function to save themes output
  await saveThemesOutput(result.text, structuredThemes);

  console.log("Themes identified and saved successfully.");
  return { textThemes: result.text, structuredThemes };
}

// Generate structured theme output
async function generateStructuredThemes(model, text) {
  const structuredLLM = model.withStructuredOutput(themeSchema, {
    strict: true,
  });
  const structuredThemePrompt = PromptTemplate.fromTemplate(`
Generate a JSON output based on the following identified themes:

{text}

Create a structured output with an array of themes, each containing:
1. title: A short title summarizing the theme.
2. description: A brief summary explaining what is said about this theme.
3. subThemes: An array of sub-topics related to the main theme. If there are no sub-themes, provide an empty array.

Ensure that every theme has all three fields, even if subThemes is an empty array.
  `);
  return await structuredLLM.invoke(
    await structuredThemePrompt.formatPromptValue({ text })
  );
}

// Part 3: Summary Generation
async function generateSummary(docs) {
  console.log("Starting summary generation...");

  const model = initializeOpenAIModel();
  const chain = createSummarizationChain(model);

  console.log("Generating summary...");
  const result = await chain.invoke({ input_documents: docs });

  const structuredOutput = await generateStructuredOutput(model, result.text);
  console.log(structuredOutput);

  await saveOutput(structuredOutput, result.text);

  console.log("Summary generation completed successfully.");
  return { structuredOutput, textOutput: result.text };
}

// Initialize OpenAI model
function initializeOpenAIModel() {
  console.log("Initializing OpenAI model...");
  return new ChatOpenAI({
    temperature: 0.3,
    modelName: "gpt-4o-mini-2024-07-18",
    maxTokens: 4000,
  });
}

// Create summarization chain
function createSummarizationChain(model) {
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

  return loadSummarizationChain(model, {
    type: "map_reduce",
    mapPrompt: mapPrompt,
    combinePrompt: combinePrompt,
  });
}

// Generate structured output
async function generateStructuredOutput(model, text) {
  console.log("Generating structured output...");
  const structuredLLM = model.withStructuredOutput(outputSchema, {
    strict: true,
  });
  const structuredOutputPrompt = PromptTemplate.fromTemplate(`
Generate a JSON output based on the following text:
{text}
  `);
  return await structuredLLM.invoke(
    await structuredOutputPrompt.formatPromptValue({ text })
  );
}

// Save output
async function saveOutput(structuredOutput, textOutput) {
  console.log("Saving the summary as JSON and text...");
  const jsonOutputPath = path.join(__dirname, "output", "summary.json");
  await fs.writeFile(jsonOutputPath, JSON.stringify(structuredOutput, null, 2));
  console.log(`JSON summary saved to ${jsonOutputPath}`);

  const textOutputPath = path.join(__dirname, "output", "summary.md");
  await fs.writeFile(textOutputPath, textOutput);
  console.log(`Text summary saved to ${textOutputPath}`);
}

// New function to save themes output
async function saveThemesOutput(textThemes, structuredThemes) {
  console.log("Saving themes output...");
  const themesTextOutputPath = path.join(__dirname, "output", "themes.md");
  await fs.writeFile(themesTextOutputPath, textThemes);
  console.log(`Text themes saved to ${themesTextOutputPath}`);

  const themesJsonOutputPath = path.join(__dirname, "output", "themes.json");
  await fs.writeFile(themesJsonOutputPath, JSON.stringify(structuredThemes, null, 2));
  console.log(`JSON themes saved to ${themesJsonOutputPath}`);
}

// Main function
async function summarizeTranscript() {
  try {
    const transcriptPath = path.join(__dirname, "transcripts", "flomodia-thibaud-elziere", "openai-whisper.txt");
    
    // Part 1: Process the transcript
    const docs = await processTranscript(transcriptPath);
    
    // Part 2: Identify themes
    const { textThemes, structuredThemes } = await identifyThemes(docs);
    
    // Part 3: Generate the summary
    const { structuredOutput, textOutput } = await generateSummary(docs);

    console.log("Transcript summarization completed successfully.");
  } catch (error) {
    console.error("Error summarizing transcript:", error);
  }
}

summarizeTranscript();