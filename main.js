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

// Define predefined sections with descriptions
const predefinedSections = {
  summary: "Brief overview capturing the essence of the entire video content.",
  key_points: "Consolidated list of main topics or concepts discussed.",
  detailed_notes: "Comprehensive and structured summary of the video content, including key arguments, examples, and explanations.",
  important_quotes: "List of the most notable quotes or standout statements.",
  actions_takeaways: "Compiled list of practical tips, steps, or lessons viewers can apply.",
  glossary: "Definitions of important specialized terms or concepts introduced.",
  references_and_resources: "Any external resources or citations mentioned."
};

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
async function generateSummary(docs, language = 'English', focusedThemes = [], selectedSections = []) {
  console.log("Starting summary generation...");

  const model = initializeOpenAIModel();
  const chain = createSummarizationChain(model, language, focusedThemes, selectedSections);

  console.log("Generating summary...");
  const result = await chain.invoke({ input_documents: docs });

  const structuredOutput = await generateStructuredOutput(model, result.text, language, selectedSections);
  console.log(structuredOutput);

  // Create personalization parameters object
  const personalizationParams = {
    language,
    focusedThemes,
    selectedSections
  };

  // Save outputs including personalization parameters
  await saveOutput(structuredOutput, result.text, personalizationParams);

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
function createSummarizationChain(model, language, focusedThemes, selectedSections) {
  const sectionDescriptions = selectedSections.map(section => 
    `- ${section}: ${predefinedSections[section]}`
  ).join('\n');

  const mapPrompt = PromptTemplate.fromTemplate(`
Analyze the following part of a video transcript and create a partial summary. Focus on the content of this specific part.
Generate the summary in ${language}.

# Focus Themes:
${focusedThemes.length > 0 ? focusedThemes.join(', ') : 'All themes'}

# Summary Sections:
${sectionDescriptions}

# Transcript Part:
{text}

Provide a concise summary focusing on the specified themes and sections:
  `);

  const combinePrompt = PromptTemplate.fromTemplate(`
Create a comprehensive cheatsheet for the entire video content by synthesizing the following partial summaries. 
Organize the information logically and eliminate redundancies.
Generate the cheatsheet in ${language}.

# Focus Themes:
${focusedThemes.length > 0 ? focusedThemes.join(', ') : 'All themes'}

# Summary Sections:
${sectionDescriptions}

{text}

Generate a final cheatsheet with these sections, ensuring each section adheres to its description:
${selectedSections.join(', ')}

Ensure the final cheatsheet is well-organized, covers the entire video content, and focuses on the specified themes:
  `);

  return loadSummarizationChain(model, {
    type: "map_reduce",
    mapPrompt: mapPrompt,
    combinePrompt: combinePrompt,
  });
}

// Generate structured output
async function generateStructuredOutput(model, text, language, selectedSections) {
  console.log("Generating structured output...");
  
  // Create a new schema based on selected sections
  const dynamicSchema = z.object(
    Object.fromEntries(
      selectedSections.map(section => {
        return [section, section === 'summary' ? z.string() : z.array(z.string())];
      })
    )
  );

  const sectionDescriptions = selectedSections.map(section => 
    `- ${section}: ${predefinedSections[section]}`
  ).join('\n');

  const structuredLLM = model.withStructuredOutput(dynamicSchema, {
    strict: true,
  });
  const structuredOutputPrompt = PromptTemplate.fromTemplate(`
Generate a JSON output based on the following text in ${language}:
{text}

Include only the following sections, adhering to their descriptions:
${sectionDescriptions}
  `);
  return await structuredLLM.invoke(
    await structuredOutputPrompt.formatPromptValue({ text })
  );
}

// Save output
async function saveOutput(structuredOutput, textOutput, personalizationParams) {
  console.log("Saving the summary and personalization parameters...");
  
  const outputDir = path.join(__dirname, "output");
  
  // Ensure the output directory exists
  await fs.mkdir(outputDir, { recursive: true });

  // Save JSON summary
  const jsonOutputPath = path.join(outputDir, "summary.json");
  await fs.writeFile(jsonOutputPath, JSON.stringify(structuredOutput, null, 2));
  console.log(`JSON summary saved to ${jsonOutputPath}`);

  // Save text summary
  const textOutputPath = path.join(outputDir, "summary.md");
  await fs.writeFile(textOutputPath, textOutput);
  console.log(`Text summary saved to ${textOutputPath}`);

  // Save personalization parameters
  const paramsOutputPath = path.join(outputDir, "personalization_params.json");
  await fs.writeFile(paramsOutputPath, JSON.stringify(personalizationParams, null, 2));
  console.log(`Personalization parameters saved to ${paramsOutputPath}`);
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

// Add this function to randomly select elements from an array
function randomSample(array, n) {
  const shuffled = array.slice().sort(() => 0.5 - Math.random());
  return shuffled.slice(0, n);
}

// Main function
async function summarizeTranscript() {
  try {
    const transcriptPath = path.join(__dirname, "transcripts", "flomodia-thibaud-elziere", "openai-whisper.txt");
    
    // Hardcoded personalization parameters (to be replaced with user inputs later)
    const language = 'English';
    const numberOfThemesToFocus = 2; // You can adjust this number as needed
    const selectedSections = ['summary', 'key_points', 'detailed_notes', 'actions_takeaways'];

    // Part 1: Process the transcript
    const docs = await processTranscript(transcriptPath);
    
    // Part 2: Identify themes
    const { textThemes, structuredThemes } = await identifyThemes(docs);
    
    // Randomly select focused themes from the detected themes
    const allThemes = structuredThemes.themes.map(theme => theme.title);
    const focusedThemes = randomSample(allThemes, Math.min(numberOfThemesToFocus, allThemes.length));
    console.log("Randomly selected focused themes:", focusedThemes);
    
    // Part 3: Generate the summary with personalization
    const { structuredOutput, textOutput } = await generateSummary(docs, language, focusedThemes, selectedSections);

    console.log("Transcript summarization completed successfully.");
  } catch (error) {
    console.error("Error summarizing transcript:", error);
  }
}

summarizeTranscript();