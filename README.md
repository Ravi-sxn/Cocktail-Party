# Cocktail-Party
Speech-Separation 
1. Run Pre_process_Mix_Sound   with one of file in Input_Audios
2. Run RAVI_MAIN.ipynb  input a o/p dir name as you want eg RAVI-1 etc then if u want unlearn any voice then enter 1/2/3 or if you dont want to unlearnn any voice  just press enter
If you want to Fine Tune it on any dataset like LibriCSS/ESC-50  use Fine-Tune.ipynb





-------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------



This code is designed to perform speech separation for a 3-speaker mixture using the **SepFormer** model from **SpeechBrain**, evaluate separation metrics, and optionally "unlearn" (ignore) certain speakers during the combination of separated sources. The overall workflow can be broken down into several parts, which I will explain in detail below.

Key Parts of the Code:

1. Imports and Dependencies:
   - torchaudio: Used for loading and saving audio files.
   - torch: The main framework used to manipulate the waveform and perform tensor-based operations.
   - speechbrain.pretrained: Imports the pre-trained SepFormerSeparation** model from SpeechBrain for speech separation.
   - IPython.display: Provides an interactive method to play audio directly in Jupyter notebooks.
   - mir_eval.separation: For calculating separation performance metrics such as SDR, SAR.
   - pystoi: For computing the STOI (Short-Time Objective Intelligibility) score, which evaluates speech quality.
   - sklearn.metrics: For calculating **mutual information, which measures the dependence between two signals.
   - numpy: For general numerical operations like logarithmic transformations.


2. Waveform Plotting (plot_waveform):
   This function visualizes the waveform of the audio. It creates a plot where the x-axis represents time (in samples) and the y-axis represents the amplitude of the  waveform. This helps to visualize the audio signal before and after processing.

3. Audio Playback (play_audio):
   This function uses **IPython.display.Audio** to play audio from a file directly in a Jupyter notebook or IPython environment.

4. Utility Functions:
   - match_length: Ensures that the reference and estimated audio signals are of the same length by trimming or padding them. This is necessary for performing metric calculations where both waveforms need to align in size.
   - mutual_information: Computes the mutual information score between the reference and estimated waveforms. Mutual information measures the amount of information shared between the two signals and can be used as a measure of similarity.
   - spectral_coherence_index: Computes the **Spectral Coherence Index (SCI)**, which measures the alignment of the frequency content of the reference and estimated signals. This helps assess the similarity between two signals in the frequency domain.
   - recursive_spectral_grouping: This method performs spectral grouping using Singular Value Decomposition (SVD) to extract the most significant components from the magnitude of the Short-Time Fourier Transform (STFT) of the waveform. It helps with separating the sources in the audio.
   - iterative_refinement: A process that refines the estimate of the separated signal iteratively. It tries to improve the separation by focusing on residuals and refining them iteratively, based on SVD.

5. Metrics Calculation (evaluate_metrics):
   This function computes the following metrics for each separated source:
   - **SDR (Signal-to-Distortion Ratio)**: Measures the quality of the separation in dB. Higher SDR indicates better separation.
   - **SAR (Signal-to-Artifact Ratio)**: Evaluates how much artifact noise is present in the separation. Higher SAR means fewer artifacts.
   - **STOI (Short-Time Objective Intelligibility)**: Measures the intelligibility of the separated speech. Higher STOI values indicate clearer speech.
   - **MI (Mutual Information)**: A metric that quantifies the similarity between the reference and separated signals.
   - **SCI (Spectral Coherence Index)**: Measures the spectral alignment between the reference and separated signals.

6. **Plotting Metrics (`plot_metrics`)**:
   This function generates bar plots for each of the metrics (`SDR`, `SAR`, `STOI`, `MI`, `SCI`) for each speaker. These plots give a visual representation of the performance of the separation for each speaker.

7. **Speech Separation and Unlearning (`separate_and_unlearn`)**:
   This is the main function that:
   - Loads the pre-trained **SepFormer** model for 3-speaker separation from SpeechBrain.
   - Loads the input audio file using **torchaudio**.
   - If the input mixture is stereo, it averages the channels into a mono signal.
   - Visualizes the waveform of the input mixture.
   - Performs the separation using **SepFormer** and saves each separated speaker’s audio to an output directory.
   - Evaluates the separation quality for each speaker and stores the metrics.
   - Optionally combines the separated sources while ignoring (unlearning) specific speakers (as specified by the user). For example, if the user doesn't want a certain speaker’s audio to be part of the final mixed output, that speaker's separated audio will be excluded.
   - Finally, it plays the separated audio files and the mixed output without the unlearned speakers.

### Workflow

1. **Load and Pre-process Input:**
   - Load the input audio file.
   - Check if the audio has more than one channel (i.e., stereo) and convert it to mono if necessary.
   - Plot and play the waveform of the mixture.

2. **Speech Separation:**
   - Use **SepFormer** to separate the 3 sources (speakers) from the input mixture. The model outputs a tensor containing the separated sources.

3. **Unlearning Speakers:**
   - Optionally exclude specific speakers from the final mixed output. The `unlearn_speakers` parameter is a list of speaker indices that should be ignored when combining the separated sources.

4. **Evaluation:**
   - For each separated source, calculate the separation metrics (SDR, SAR, STOI, MI, SCI).
   - Save each separated source as a `.wav` file in the specified output directory.

5. **Visualization and Playback:**
   - Plot the waveforms of the separated sources and play the separated audio files.
   - If the user decides to combine the sources excluding certain speakers, save this final output as a `.wav` file and play it.

6. **Metrics Plotting:**
   - After processing, a bar plot is generated showing the evaluation metrics for each speaker, providing insight into the performance of the separation.

### Entry Point (Main Code Execution)

- When the script is run, it prompts the user for the input audio path and output directory.
- It then asks the user which speakers to "unlearn" (exclude from the final mixed output).
- The function `separate_and_unlearn` is called to execute the whole process.

### Example Usage

Assume you have an input audio file with three overlapping speakers. After running the script, the program:
1. Separates the voices.
2. Optionally "unlearns" a speaker (excludes their voice from the final mixed output).
3. Saves the separated voices as `.wav` files.
4. Plots the metrics for each speaker (SDR, SAR, STOI, MI, SCI).
5. Plays the separated voices and the combined output (excluding any unlearned speakers).

This approach is particularly useful in scenarios where you want to separate audio signals from a noisy or overlapping mixture of multiple speakers, evaluate the quality of separation, and even adjust the final mixed output by removing specific sources.
