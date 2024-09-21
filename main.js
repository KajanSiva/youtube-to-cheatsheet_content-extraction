import { ChatOpenAI } from "@langchain/openai";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from 'url';
import 'dotenv/config';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function generateSummarizationPrompt(transcript) {
  return `
Analyze the following video transcript and create a cheatsheet that includes the specified sections. Ensure that all content is solely derived from the transcript without adding any external information or assumptions.
The output must be in the language of the transcript.

---

# Specific instructions for the "Detailed Notes" section:
Provide a comprehensive and structured summary of the video content. Include key arguments, examples, anecdotes, and explanations given by the presenter. Use headings and bullet points to organize the information. This section should expand on the points mentioned in "Key Points/Main Ideas" by offering more context and details.

---

# Cheatsheet Sections:
- Summary: Provide a brief overview capturing the essence of the video content.
- Key Points/Main Ideas: List the main topics or concepts discussed in the video as bullet points.
- Detailed Notes: [Use the instructions above for this section.]
- Important Quotes or Statements: Extract notable quotes or standout statements from the video.
- Actions/Takeaways: List practical tips, steps, or lessons that viewers can apply.
- Glossary/Definitions: Define any specialized terms or concepts introduced in the video.
- References and Resources: Include any external resources or citations mentioned in the video.

---

# Transcript:
${transcript}
`;
}

async function summarizeTranscript() {
  try {
    console.log("Starting transcript summarization...");

    // Read the transcript file
    console.log("Reading transcript file...");
    const transcript = await fs.readFile(
      path.join(__dirname, "transcripts", "flomodia-thibaud-elziere", "openai-whisper.txt"),
      "utf-8"
    );
    console.log("Transcript file read successfully.");

    // Initialize the OpenAI model
    console.log("Initializing OpenAI model...");
    const model = new ChatOpenAI({
      temperature: 0.3,
      modelName: "gpt-4o-2024-08-06",
    });
    console.log("OpenAI model initialized.");

    // Generate the prompt directly using the existing function
    console.log("Generating summarization prompt...");
    const prompt = generateSummarizationPrompt(transcript);
    console.log("Summarization prompt generated.");

    // Call the LLM
    console.log("Calling the language model...");
    const result = await model.invoke(prompt);
    console.log("Language model response received.");

    // Save the result
    console.log("Saving the summary...");
    const outputPath = path.join(__dirname, "output", "summary.md");
    await fs.writeFile(outputPath, result.content);
    console.log(`Summary saved to ${outputPath}`);

    console.log("Transcript summarization completed successfully.");
  } catch (error) {
    console.error("Error summarizing transcript:", error);
  }
}

summarizeTranscript();