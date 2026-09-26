"""
Example script demonstrating JailbreakBench Judge usage

This script shows how to use the JailbreakJudge class to evaluate
jailbreak attack results following the 4-step process:

1. Initialize the Results Data Structure
2. Set Up the Automated Judge  
3. Extract and Calculate Metrics
4. Access Detailed Reasoning Logs
"""

from jailbreak_judge import JailbreakJudge, JudgeMetrics
import glob
import os

def main():
    """
    Main execution function demonstrating judge usage.
    """
    print("=" * 80)
    print("JailbreakBench Judge Evaluation")
    print("=" * 80)
    
    # Configuration
    USE_REAL_JUDGE = False  # Set to True if you have API access
    API_KEY = None  # Set your API key if using real judge
    
    # Step 1 & 2: Initialize judge (this sets up the automated judge)
    print("\n[Step 1-2] Initializing Judge...")
    judge = JailbreakJudge(
        api_key=API_KEY,
        use_mock=not USE_REAL_JUDGE
    )
    
    # Find CSV files in current directory
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("\nNo CSV files found in current directory.")
        print("Please ensure your attack result CSV files are in this directory.")
        print("Expected columns: goal, prompt, llm_response, [category], [index]")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s):")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"  {i}. {csv_file}")
    
    # Evaluate all CSV files
    all_metrics = []
    
    for csv_file in csv_files:
        try:
            print(f"\n{'='*80}")
            print(f"Evaluating: {csv_file}")
            print(f"{'='*80}\n")
            
            # Step 1-3: Evaluate CSV and calculate metrics
            metrics = judge.evaluate_csv(csv_file, batch_size=10)
            
            # Display results
            judge.print_metrics(metrics)
            
            # Save results
            output_prefix = os.path.splitext(csv_file)[0] + "_judge"
            judge.save_results(metrics, output_dir=".", prefix=output_prefix)
            
            all_metrics.append({
                'file': csv_file,
                'metrics': metrics
            })
            
        except Exception as e:
            print(f"Error evaluating {csv_file}: {e}")
            continue
    
    # Step 4: Access and display reasoning logs
    print(f"\n{'='*80}")
    print("Step 4: Accessing Detailed Reasoning Logs")
    print(f"{'='*80}\n")
    
    reasoning_logs = judge.get_reasoning_logs()
    print(f"Total reasoning logs collected: {len(reasoning_logs)}")
    
    if reasoning_logs:
        print("\nSample reasoning log (first entry):")
        print("-" * 80)
        sample_log = reasoning_logs[0]
        print(f"Index: {sample_log['index']}")
        print(f"Goal: {sample_log['goal']}")
        print(f"Is Jailbroken: {sample_log['is_jailbroken']}")
        print(f"Is Refusal: {sample_log['is_refusal']}")
        print(f"Reasoning: {sample_log['reasoning']}")
        print(f"Response Preview: {sample_log['response_preview']}")
    
    # Summary across all files
    if len(all_metrics) > 1:
        print(f"\n{'='*80}")
        print("Summary Across All Files")
        print(f"{'='*80}\n")
        
        total_attempts = sum(m['metrics'].total_attempts for m in all_metrics)
        total_jailbreaks = sum(m['metrics'].successful_jailbreaks for m in all_metrics)
        total_refusals = sum(m['metrics'].refusals for m in all_metrics)
        
        avg_asr = (total_jailbreaks / total_attempts * 100) if total_attempts > 0 else 0
        avg_refusal = (total_refusals / total_attempts * 100) if total_attempts > 0 else 0
        
        print(f"Total Files Evaluated:       {len(all_metrics)}")
        print(f"Total Attempts:              {total_attempts}")
        print(f"Total Successful Jailbreaks: {total_jailbreaks}")
        print(f"Total Refusals:              {total_refusals}")
        print(f"Average Attack Success Rate: {avg_asr:.2f}%")
        print(f"Average Refusal Rate:        {avg_refusal:.2f}%")
        
        print("\nPer-File Breakdown:")
        for item in all_metrics:
            m = item['metrics']
            print(f"\n  {item['file']}:")
            print(f"    ASR: {m.attack_success_rate:.2f}%")
            print(f"    Refusal Rate: {m.refusal_rate:.2f}%")
    
    print(f"\n{'='*80}")
    print("Evaluation Complete!")
    print(f"{'='*80}\n")


def evaluate_single_csv(csv_path: str, api_key: Optional[str] = None, use_mock: bool = True):
    """
    Convenience function to evaluate a single CSV file.
    
    Args:
        csv_path: Path to CSV file
        api_key: API key for judge (if using real judge)
        use_mock: Whether to use mock judge
        
    Returns:
        JudgeMetrics object
    """
    judge = JailbreakJudge(api_key=api_key, use_mock=use_mock)
    metrics = judge.evaluate_csv(csv_path)
    judge.print_metrics(metrics)
    
    output_prefix = os.path.splitext(csv_path)[0] + "_judge"
    judge.save_results(metrics, prefix=output_prefix)
    
    return metrics


if __name__ == "__main__":
    from typing import Optional
    main()
