#!/bin/bash
# Complete pipeline for all experiments

echo "================================================"
echo "    NEURAL OPERATOR ROBUSTNESS EXPERIMENTS     "
echo "================================================"

# Function to run experiment
run_experiment() {
    local exp_name=$1
    echo ""
    echo "Running $exp_name experiment..."
    echo "--------------------------------"
    
    cd experiments/$exp_name
    
    # Train models
    echo "→ Training models..."
    python train.py
    
    # Evaluate noise robustness
    echo "→ Evaluating noise robustness..."
    python evaluate.py
    
    # Generate figures
    echo "→ Generating figures..."
    python visualize.py
    
    cd ../..
    echo "✓ $exp_name complete!"
}

# Run all experiments
# for exp in naca darcy burgers; do
for exp in naca; do
    run_experiment $exp
done

echo ""
echo "================================================"
echo "✅ ALL EXPERIMENTS COMPLETE!"
echo "   Results in: results/"
echo "================================================"