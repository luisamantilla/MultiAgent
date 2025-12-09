"""
Web-based Human Review Interface for LLM Evaluation Results

This Flask app provides a user-friendly web interface for human reviewers
to evaluate LLM-generated step matches from biological process evaluations.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
from datetime import datetime
from pathlib import Path
import argparse


app = Flask(__name__)

# Global variables to store evaluation data
evaluation_data = None
evaluation_file_path = None
reviewable_steps = []
review_session = {
    "start_time": datetime.now().isoformat(),
    "reviewer": None,
    "total_reviews": 0,
    "good_matches": 0,
    "bad_matches": 0,
    "skipped": 0
}


def load_evaluation_data(file_path: str):
    """Load LLM evaluation results from JSON file"""
    global evaluation_data, evaluation_file_path, reviewable_steps
    
    evaluation_file_path = file_path
    with open(file_path, 'r') as f:
        evaluation_data = json.load(f)
    
    # Extract reviewable steps (those with LLM matches)
    reviewable_steps = [
        step for step in evaluation_data.get("step_level_results", [])
        if step.get("matched_prediction") is not None
    ]
    
    return evaluation_data


@app.route('/')
def index():
    """Main review interface"""
    if evaluation_data is None:
        return "No evaluation data loaded. Please specify an evaluation file."
    
    return render_template('review_interface.html', 
                         evaluation_data=evaluation_data,
                         reviewable_steps=reviewable_steps,
                         total_steps=len(reviewable_steps))


@app.route('/api/step/<int:step_index>')
def get_step(step_index):
    """Get step data for review"""
    if 0 <= step_index < len(reviewable_steps):
        step = reviewable_steps[step_index]
        
        # Add context about the biological process
        context_info = {
            "evaluation_mode": evaluation_data.get("evaluation_mode", "unknown"),
            "model": evaluation_data.get("model", "unknown"),
            "context": evaluation_data.get("context", ""),
            "total_steps": len(reviewable_steps),
            "step_index": step_index
        }
        
        return jsonify({"step": step, "context": context_info})
    
    return jsonify({"error": "Invalid step index"}), 400


@app.route('/api/submit_review', methods=['POST'])
def submit_review():
    """Submit human review judgment"""
    global review_session
    
    data = request.json
    step_index = data.get('step_index')
    judgment = data.get('judgment')  # 'good', 'bad', 'skip'
    reviewer = data.get('reviewer', 'Anonymous')
    
    if 0 <= step_index < len(reviewable_steps):
        # Update step with human judgment
        step = reviewable_steps[step_index]
        step['human_judgment'] = judgment
        step['human_review_timestamp'] = datetime.now().isoformat()
        step['human_reviewer'] = reviewer
        
        # Update session stats
        review_session["reviewer"] = reviewer
        review_session["total_reviews"] += 1
        
        if judgment == 'good':
            review_session["good_matches"] += 1
        elif judgment == 'bad':
            review_session["bad_matches"] += 1
        elif judgment == 'skip':
            review_session["skipped"] += 1
        
        # Update evaluation data
        for original_step in evaluation_data["step_level_results"]:
            if (original_step.get("step_detail") == step.get("step_detail") and 
                original_step.get("process_name") == step.get("process_name")):
                original_step.update(step)
                break
        
        return jsonify({"success": True, "session_stats": review_session})
    
    return jsonify({"error": "Invalid step index"}), 400


@app.route('/api/save_progress')
def save_progress():
    """Save current review progress to file"""
    if evaluation_data is None or evaluation_file_path is None:
        return jsonify({"error": "No evaluation data to save"}), 400
    
    # Update session info
    review_session["end_time"] = datetime.now().isoformat()
    evaluation_data["human_review_session"] = review_session
    
    # Create output filename
    base_name = os.path.splitext(evaluation_file_path)[0]
    output_file = f"{base_name}_human_reviewed.json"
    
    # Save updated data
    with open(output_file, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    return jsonify({
        "success": True, 
        "output_file": output_file,
        "session_stats": review_session
    })


@app.route('/api/session_stats')
def get_session_stats():
    """Get current session statistics"""
    total = review_session["total_reviews"]
    stats = dict(review_session)
    
    if total > 0:
        stats["good_percentage"] = (review_session["good_matches"] / total) * 100
        stats["bad_percentage"] = (review_session["bad_matches"] / total) * 100
        stats["skip_percentage"] = (review_session["skipped"] / total) * 100
        
        # Calculate human-validated precision
        stats["human_precision"] = review_session["good_matches"] / len(reviewable_steps)
    
    stats["completion_percentage"] = (total / len(reviewable_steps)) * 100 if reviewable_steps else 0
    
    return jsonify(stats)


def create_html_template():
    """Create the HTML template for the review interface"""
    template_dir = Path(__file__).parent / "templates"
    template_dir.mkdir(exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧬 Biological Process Review Interface</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .header h1 {
            margin: 0;
            font-size: 2em;
        }
        
        .stats-panel {
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }
        
        .stat-item {
            text-align: center;
            min-width: 120px;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .reviewer-input {
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .review-card {
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 1000px;
            margin: 0 auto 20px auto;
        }
        
        .step-header {
            border-bottom: 2px solid #eee;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
        
        .step-number {
            font-size: 1.2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .process-name {
            font-size: 1.1em;
            color: #555;
            margin: 5px 0;
        }
        
        .content-section {
            margin: 20px 0;
            padding: 15px;
            border-radius: 8px;
        }
        
        .ground-truth {
            background-color: #e8f4fd;
            border-left: 4px solid #2196F3;
        }
        
        .prediction {
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
        }
        
        .reasoning {
            background-color: #f3e5f5;
            border-left: 4px solid #9c27b0;
        }
        
        .section-title {
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        
        .section-title .emoji {
            margin-right: 8px;
            font-size: 1.2em;
        }
        
        .content-text {
            line-height: 1.6;
            font-size: 1em;
        }
        
        .action-buttons {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 25px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 25px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            min-width: 120px;
        }
        
        .btn-good {
            background-color: #4CAF50;
            color: white;
        }
        
        .btn-good:hover {
            background-color: #45a049;
            transform: translateY(-2px);
        }
        
        .btn-bad {
            background-color: #f44336;
            color: white;
        }
        
        .btn-bad:hover {
            background-color: #da190b;
            transform: translateY(-2px);
        }
        
        .btn-skip {
            background-color: #ff9800;
            color: white;
        }
        
        .btn-skip:hover {
            background-color: #e68900;
            transform: translateY(-2px);
        }
        
        .btn-nav {
            background-color: #6c757d;
            color: white;
        }
        
        .btn-nav:hover {
            background-color: #5a6268;
            transform: translateY(-2px);
        }
        
        .btn-save {
            background-color: #17a2b8;
            color: white;
        }
        
        .btn-save:hover {
            background-color: #138496;
            transform: translateY(-2px);
        }
        
        .navigation {
            text-align: center;
            margin: 20px 0;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
            transition: width 0.3s ease;
        }
        
        .feedback {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            text-align: center;
        }
        
        .feedback.success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .feedback.error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .context-info {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            font-size: 0.9em;
        }
        
        @media (max-width: 768px) {
            .stats-panel {
                flex-direction: column;
                gap: 10px;
            }
            
            .action-buttons {
                flex-direction: column;
                align-items: center;
            }
            
            .btn {
                width: 200px;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 Biological Process Review Interface</h1>
        <p>Human evaluation of LLM-generated biological step matches</p>
    </div>
    
    <div class="reviewer-input">
        <label for="reviewer-name">Reviewer Name:</label>
        <input type="text" id="reviewer-name" placeholder="Enter your name" style="margin-left: 10px; padding: 5px;">
    </div>
    
    <div class="stats-panel">
        <div class="stat-item">
            <div class="stat-number" id="total-reviews">0</div>
            <div>Reviews Done</div>
        </div>
        <div class="stat-item">
            <div class="stat-number" id="good-matches">0</div>
            <div>Good Matches</div>
        </div>
        <div class="stat-item">
            <div class="stat-number" id="bad-matches">0</div>
            <div>Bad Matches</div>
        </div>
        <div class="stat-item">
            <div class="stat-number" id="completion">0%</div>
            <div>Completion</div>
        </div>
    </div>
    
    <div class="progress-bar">
        <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
    </div>
    
    <div id="feedback" class="feedback" style="display: none;"></div>
    
    <div class="review-card" id="review-card">
        <div id="step-content">
            <p>Loading evaluation data...</p>
        </div>
    </div>
    
    <div class="navigation">
        <button class="btn btn-nav" onclick="previousStep()" id="prev-btn">⬅️ Previous</button>
        <button class="btn btn-nav" onclick="nextStep()" id="next-btn">Next ➡️</button>
        <button class="btn btn-save" onclick="saveProgress()">💾 Save Progress</button>
    </div>

    <script>
        let currentStepIndex = 0;
        let totalSteps = {{ total_steps }};
        let reviewerName = '';
        
        // Load first step on page load
        window.onload = function() {
            loadStep(0);
            updateStats();
        };
        
        function loadStep(stepIndex) {
            if (stepIndex < 0 || stepIndex >= totalSteps) {
                return;
            }
            
            currentStepIndex = stepIndex;
            
            fetch(`/api/step/${stepIndex}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        showFeedback(data.error, 'error');
                        return;
                    }
                    
                    displayStep(data.step, data.context);
                    updateNavigationButtons();
                })
                .catch(error => {
                    showFeedback('Error loading step: ' + error.message, 'error');
                });
        }
        
        function displayStep(step, context) {
            const stepContent = document.getElementById('step-content');
            
            const currentJudgment = step.human_judgment || 'pending';
            const judgmentDisplay = currentJudgment !== 'pending' ? 
                `<div style="background: #e8f5e8; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                    ⚖️ <strong>Current Judgment:</strong> ${currentJudgment.toUpperCase()}
                 </div>` : '';
            
            stepContent.innerHTML = `
                <div class="step-header">
                    <div class="step-number">Step ${currentStepIndex + 1} of ${totalSteps}</div>
                    <div class="process-name">📋 Process: ${step.process_name}</div>
                </div>
                
                <div class="context-info">
                    <strong>🔬 Context:</strong> ${context.context}<br>
                    <strong>🤖 Model:</strong> ${context.model} (${context.evaluation_mode} mode)
                </div>
                
                ${judgmentDisplay}
                
                <div class="content-section ground-truth">
                    <div class="section-title">
                        <span class="emoji">🎯</span>
                        Ground Truth Step
                    </div>
                    <div class="content-text">${step.step_detail}</div>
                </div>
                
                <div class="content-section prediction">
                    <div class="section-title">
                        <span class="emoji">🤖</span>
                        LLM Matched Prediction
                    </div>
                    <div class="content-text">${step.matched_prediction}</div>
                </div>
                
                ${step.biological_reasoning ? `
                <div class="content-section reasoning">
                    <div class="section-title">
                        <span class="emoji">🧠</span>
                        LLM Reasoning
                    </div>
                    <div class="content-text">${step.biological_reasoning}</div>
                </div>
                ` : ''}
                
                <div class="action-buttons">
                    <button class="btn btn-good" onclick="submitJudgment('good')">
                        ✅ Good Match
                    </button>
                    <button class="btn btn-bad" onclick="submitJudgment('bad')">
                        ❌ Bad Match  
                    </button>
                    <button class="btn btn-skip" onclick="submitJudgment('skip')">
                        ⏭️ Skip
                    </button>
                </div>
            `;
        }
        
        function submitJudgment(judgment) {
            reviewerName = document.getElementById('reviewer-name').value || 'Anonymous';
            
            fetch('/api/submit_review', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    step_index: currentStepIndex,
                    judgment: judgment,
                    reviewer: reviewerName
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showFeedback(`Recorded as "${judgment.toUpperCase()}"`, 'success');
                    updateStats();
                    
                    // Auto-advance to next step
                    setTimeout(() => {
                        if (currentStepIndex < totalSteps - 1) {
                            nextStep();
                        } else {
                            showFeedback('🎉 All steps reviewed! Great work!', 'success');
                        }
                    }, 1500);
                } else {
                    showFeedback(data.error, 'error');
                }
            })
            .catch(error => {
                showFeedback('Error submitting review: ' + error.message, 'error');
            });
        }
        
        function previousStep() {
            if (currentStepIndex > 0) {
                loadStep(currentStepIndex - 1);
            }
        }
        
        function nextStep() {
            if (currentStepIndex < totalSteps - 1) {
                loadStep(currentStepIndex + 1);
            }
        }
        
        function updateNavigationButtons() {
            const prevBtn = document.getElementById('prev-btn');
            const nextBtn = document.getElementById('next-btn');
            
            prevBtn.disabled = currentStepIndex === 0;
            nextBtn.disabled = currentStepIndex === totalSteps - 1;
            
            prevBtn.style.opacity = prevBtn.disabled ? '0.5' : '1';
            nextBtn.style.opacity = nextBtn.disabled ? '0.5' : '1';
        }
        
        function updateStats() {
            fetch('/api/session_stats')
                .then(response => response.json())
                .then(stats => {
                    document.getElementById('total-reviews').textContent = stats.total_reviews;
                    document.getElementById('good-matches').textContent = stats.good_matches;
                    document.getElementById('bad-matches').textContent = stats.bad_matches;
                    document.getElementById('completion').textContent = 
                        Math.round(stats.completion_percentage) + '%';
                    
                    // Update progress bar
                    document.getElementById('progress-fill').style.width = 
                        stats.completion_percentage + '%';
                });
        }
        
        function saveProgress() {
            fetch('/api/save_progress')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showFeedback(`Progress saved to: ${data.output_file}`, 'success');
                    } else {
                        showFeedback(data.error, 'error');
                    }
                });
        }
        
        function showFeedback(message, type) {
            const feedback = document.getElementById('feedback');
            feedback.textContent = message;
            feedback.className = `feedback ${type}`;
            feedback.style.display = 'block';
            
            // Hide after 3 seconds
            setTimeout(() => {
                feedback.style.display = 'none';
            }, 3000);
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(event) {
            if (event.key === 'g' || event.key === 'G') {
                submitJudgment('good');
            } else if (event.key === 'b' || event.key === 'B') {
                submitJudgment('bad');
            } else if (event.key === 's' || event.key === 'S') {
                submitJudgment('skip');
            } else if (event.key === 'ArrowLeft') {
                previousStep();
            } else if (event.key === 'ArrowRight') {
                nextStep();
            }
        });
    </script>
</body>
</html>'''
    
    with open(template_dir / "review_interface.html", 'w') as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description="Web-based human review interface")
    parser.add_argument("evaluation_file", help="Path to enhanced LLM evaluation JSON file")
    parser.add_argument("--port", type=int, default=5000, help="Port to run web server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    
    args = parser.parse_args()
    
    # Load evaluation data
    try:
        load_evaluation_data(args.evaluation_file)
        print(f"Loaded evaluation data: {len(reviewable_steps)} steps to review")
    except Exception as e:
        print(f"Error loading evaluation file: {e}")
        exit(1)
    
    # Create HTML template
    create_html_template()
    
    print(f"Starting web interface at http://{args.host}:{args.port}")
    print("Use Ctrl+C to stop the server")
    
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
