import argparse
import os

# Set env vars to stop the ONNX threading errors
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["ORT_TENSORRT_FP16_ENABLE"] = "0"

import pandas as pd
from tqdm import tqdm
import librosa
import numpy as np
# Explicitly import the submodule
from speechmos import dnsmos
import wespeakerruntime as wespeaker 

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate DNSMOS scores (via SpeechMOS)')
    parser.add_argument('--results_dir', type=str, required=True, 
                        help='Path to the results directory')
    parser.add_argument('--existing_metrics', type=str, default=None,
                        help='Path to existing metrics_results.csv to merge with')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save the merged results CSV')
    parser.add_argument('--skip_merge', action='store_true',
                        help='Skip merging with existing metrics')
    parser.add_argument('--primary_model_path', type=str, default=None)
    args = parser.parse_args()
    return args

def find_audio_files(results_dir):
    audio_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == 'estimation.wav':
                file_path = os.path.join(root, file)
                path_parts = os.path.normpath(file_path).split(os.sep)
                if len(path_parts) >= 3:
                    mixture_filename = path_parts[-2] + '.wav'
                    utt_id = path_parts[-3]
                    # Also check if source.wav exists in the same directory
                    source_path = os.path.join(root, 'source.wav')
                    audio_files.append({
                        'path': file_path,
                        'source_path': source_path if os.path.exists(source_path) else None,
                        'utt_id': utt_id,
                        'mixture_filename': mixture_filename
                    })
    return audio_files

def calculate_dnsmos_scores(results_dir):
    audio_files = find_audio_files(results_dir)
    
    if len(audio_files) == 0:
        print(f"No estimation.wav files found in {results_dir}")
        return None
    
    print(f"Found {len(audio_files)} audio files to process")
    print("Using SpeechMOS (Librosa/ONNX backend)...")
    
    # Initialize WeSpeaker model for speaker embeddings
    print("Initializing WeSpeaker model for speaker similarity...")
    try:
        speaker_model = wespeaker.Speaker(lang='en')
        use_speaker_sim = True
    except Exception as e:
        print(f"Warning: Could not initialize WeSpeaker model: {e}")
        print("Continuing without speaker similarity calculation...")
        use_speaker_sim = False
    
    results = []
    
    for audio_info in tqdm(audio_files, desc="Calculating DNSMOS scores"):
        try:
            audio_path = audio_info['path']
            # Load with Librosa (returns numpy array)
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Run using the imported submodule
            scores = dnsmos.run(audio, sr=sr)
            
            result = {
                'filename': audio_info['mixture_filename'],
                'dnsmos_overall': scores['ovrl_mos'],
                'dnsmos_sig': scores['sig_mos'],
                'dnsmos_bak': scores['bak_mos'],
                'dnsmos_p808': scores['p808_mos'],
            }
            
            # Calculate speaker similarity if source.wav exists
            if use_speaker_sim and audio_info['source_path'] is not None:
                try:
                    # Extract speaker embeddings using the correct API
                    estimation_embedding = speaker_model.extract_embedding(audio_path)
                    source_embedding = speaker_model.extract_embedding(audio_info['source_path'])
                    
                    # Squeeze embeddings to 1D if needed (shape: (1, 256) -> (256,))
                    if hasattr(estimation_embedding, 'squeeze'):
                        estimation_embedding = estimation_embedding.squeeze()
                        source_embedding = source_embedding.squeeze()
                    
                    # Calculate cosine similarity using WeSpeaker's built-in method
                    cosine_score = speaker_model.compute_cosine_score(estimation_embedding, source_embedding)
                    #tqdm.write(str(cosine_score))
                    result['speaker_cosine_similarity'] = float(cosine_score)
                except Exception as e:
                    print(f"Error calculating speaker similarity for {audio_info['path']}: {e}")
                    result['speaker_cosine_similarity'] = None
            else:
                result['speaker_cosine_similarity'] = None

            #tqdm.write(str(result))
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {audio_info['path']}: {e}")
            results.append({
                'filename': audio_info['mixture_filename'],
                'dnsmos_overall': None,
                'dnsmos_sig': None,
                'dnsmos_bak': None,
                'dnsmos_p808': None,
                'speaker_cosine_similarity': None,
            })
    
    return pd.DataFrame(results)

def merge_with_existing_metrics(dnsmos_df, metrics_csv_path):
    if not os.path.exists(metrics_csv_path):
        print(f"Warning: Metrics file not found at {metrics_csv_path}")
        return dnsmos_df
    
    print(f"Loading existing metrics from: {metrics_csv_path}")
    metrics_df = pd.read_csv(metrics_csv_path)
    merged_df = pd.merge(metrics_df, dnsmos_df, on='filename', how='left')
    print(f"Merged {len(merged_df)} rows")
    return merged_df

def print_summary_statistics(results_df):
    print(f"\n{'='*70}")
    print(f"Complete Metrics Summary")
    print(f"{'='*70}")
    print(f"Total files: {len(results_df)}")
    
    metric_columns = [col for col in results_df.columns 
                     if col not in ['filename', 'alpha', 'alpha_hat'] 
                     and results_df[col].dtype in ['float64', 'float32']]
    
    for metric in metric_columns:
        mean_score = results_df[metric].mean()
        std_score = results_df[metric].std()
        metric_display = metric.replace('_', ' ').upper()
        print(f"{metric_display:20s}: {mean_score:.4f} ± {std_score:.4f}")
    
    print(f"{'='*70}\n")

def main():
    args = parse_args()
    
    print(f"Processing audio files in: {args.results_dir}")
    dnsmos_df = calculate_dnsmos_scores(args.results_dir)
    
    if dnsmos_df is None or len(dnsmos_df) == 0:
        print("No results to save.")
        return
    
    if not args.skip_merge:
        if args.existing_metrics:
            metrics_csv_path = args.existing_metrics
        else:
            metrics_csv_path = os.path.join(args.results_dir, 'metrics_results.csv')
        
        if os.path.exists(metrics_csv_path):
            final_df = merge_with_existing_metrics(dnsmos_df, metrics_csv_path)
        else:
            print(f"No existing metrics found at {metrics_csv_path}, using DNSMOS only")
            final_df = dnsmos_df
    else:
        final_df = dnsmos_df
    
    if args.output_csv:
        output_csv_path = args.output_csv
    else:
        if args.skip_merge:
            output_csv_path = os.path.join(args.results_dir, 'dnsmos_results.csv')
        else:
            output_csv_path = os.path.join(args.results_dir, 'metrics_with_dnsmos.csv')
    
    final_df.to_csv(output_csv_path, index=False)
    print(f"\nSaved results to: {output_csv_path}")
    print_summary_statistics(final_df)

if __name__ == '__main__':
    main()