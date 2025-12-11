# 📘 How to Use IDE_prompt.md

## 🎯 Purpose

`IDE_prompt.md` is a **universal template** for instructing an AI IDE to create comprehensive learning guides for deep learning repositories. It was distilled from your excellent FlashOCC guide.

## 🚀 Quick Start

### Step 1: Copy the Prompt

When starting analysis of a new repo (e.g., BEVFormer, Dreamer, RT-2), paste this:

```
I want to create a complete mastery guide for this repository.

Please follow the template in @IDE_prompt.md exactly.

Domain: [Choose: Autonomous Driving / Embodied AI / VLA / World Model / End-to-End]
Target time: 4-6 hours of focused learning
Depth: Implementation-ready understanding

Start with Chapter 0: Complete inheritance chain analysis.

IMPORTANT: Follow ALL self-verification protocols. Verify every claim with code citations.
```

### Step 2: Monitor Progress

The AI should:
1. ✅ Create structured docs in `testwl/` directory
2. ✅ Follow 3-round revision process
3. ✅ Request PDF analysis when needed (you'll help with external tool)
4. ✅ Self-correct errors proactively

### Step 3: Quality Check

After AI completes, verify:
- [ ] All file paths and line numbers are accurate
- [ ] Numerical examples calculate correctly
- [ ] No generic statements (everything repo-specific)
- [ ] Self-assessment questions exist with answers
- [ ] Comparison doc compares with 2-4 related methods

## 📂 Expected Output Structure

```
testwl/
├── [ProjectName]_Complete_Mastery_Guide.md  # Main learning guide (6000+ lines)
├── Algorithm_Comparison.md                   # vs BEVDet, BEVFormer, etc.
├── Version_Evolution.md                      # V1→V2→V3 changes (if applicable)
├── Quick_Reference.md                        # Config params, API reference
├── FAQ_and_Pitfalls.md                      # Common errors & solutions
└── Self_Assessment.md                       # 30 questions with answers
```

## 🎓 What Makes This Template Special

### 1. **Anti-Hallucination Enforcement**
- Every claim must cite `file.py:line`
- Every shape must be verified with concrete dimensions
- Red flags: "probably", "seems to", "likely"
- Fix: Mark uncertain → verify → update or remove

### 2. **Domain-Specific Checklists**
- **Autonomous Driving**: Coordinate systems, BEV, multi-camera, depth
- **Embodied AI**: Action/observation spaces, policy, sim2real
- **VLA**: Multi-modal fusion, instruction grounding, action decoding
- **World Models**: State representation, dynamics, planning
- **End-to-End**: Input/output spaces, training paradigm, safety

### 3. **Pedagogical Structure**
- Teacher persona explanations (WHY before WHAT)
- Time estimates (⏱️) and difficulty (⭐⭐⭐)
- Self-check questions after each major section
- Numerical examples with concrete values
- Memory aids and analogies

### 4. **Comparison Framework** (横向对比)
- Side-by-side tables with 2-4 related methods
- "When to use each" guidelines
- Evolution timeline (mermaid diagram)
- Key insights and lessons

### 5. **Version Evolution** (纵向对比)
- What changed V1→V2→V3
- Why each change (performance/simplicity/deployment)
- Migration guide for users

### 6. **Homework System** (课后作业)
- 30 questions across 3 difficulty levels
- Architecture (10) + Algorithm (10) + Implementation (10)
- Answer key with detailed explanations
- Passing score: 80%

## 🔄 The 3-Round Process

### Round 1: First Draft
- AI writes all chapters 0-5
- Includes code citations (rough OK)
- Basic diagrams and one numerical example per algorithm

### Round 2: Accuracy Sweep
- AI re-verifies EVERY code citation
- Re-calculates EVERY shape
- Searches codebase for EVERY claimed feature
- Marks verified claims with `⚠️ 已验证`

### Round 3: Pedagogical Enhancement
- Adds teacher explanations
- Inserts self-check questions
- Improves diagrams
- Completes homework and FAQ docs

## 📖 PDF Analysis Workflow

When AI encounters papers it needs to analyze:

**AI will stop and say**:
```
⚠️ PDF Analysis Required

I need: [Paper name]
Focus: [Specific sections]
Information needed: [What to extract]

Please use external AI tool to extract and paste here.
```

**You do**:
1. Copy paper content to Claude/GPT/Kimi
2. Ask: "Extract algorithm sections, math formulas, and key differences from code"
3. Paste response back to IDE
4. AI continues comparison (paper vs code)

## ✅ Quality Indicators

### Good Signs
- ✅ Every algorithm has: Math + Code (file:line) + Numerical example
- ✅ Shapes verified at every step (B=2, N=6, H=128, W=352 → ...)
- ✅ Comparison table populated with real numbers
- ✅ Self-assessment has 30 questions with answer key
- ✅ No vague language ("probably", "seems to")

### Red Flags
- ❌ Generic descriptions ("extracts features", "does fusion")
- ❌ Missing line numbers or wrong file paths
- ❌ Shapes don't match dimensions (e.g., 4x4 matrix multiplied by 3D vector)
- ❌ Claims without code evidence
- ❌ Comparison with "unknown" or "N/A" values

## 🛠️ Customization Tips

### For Your Specific Needs

**If you want MORE depth on certain topics**:
```
In Chapter 2 (Algorithm Deep Dive), spend extra time on:
- [Algorithm X]: Include 3 numerical examples instead of 1
- [Algorithm Y]: Add pseudocode walkthrough
```

**If you want LESS on certain parts**:
```
Chapter 4 (Data Pipeline) can be shorter since I'm familiar with data loading.
Focus on repo-specific augmentation techniques only.
```

**If you want specific comparisons**:
```
In Algorithm_Comparison.md, must compare with:
1. [Method A] - focus on architectural differences
2. [Method B] - focus on performance tradeoffs
3. [Method C] - focus on implementation complexity
```

## 📊 Success Metrics

After using this template 3-5 times, you should have:

### Structured Knowledge Base
```
my_research/
├── FlashOCC/
│   └── testwl/  [6 files, ~10K lines]
├── BEVFormer/
│   └── testwl/  [6 files, ~8K lines]
├── Dreamer/
│   └── testwl/  [6 files, ~7K lines]
└── RT-2/
    └── testwl/  [6 files, ~9K lines]
```

### Easy Comparison
- Open 2 repos' Algorithm_Comparison.md side-by-side
- Instantly see: BEVFormer uses transformer attention, FlashOCC uses LSS
- Decision: Pick BEVFormer for accuracy, FlashOCC for speed

### Rapid Mastery
- New repo: Read 4-6 hours → complete Self_Assessment → 80%+ score
- Old skill: You now understand core ideas without weeks of struggle
- New skill: You can modify/extend code confidently

## 🎯 Advanced Usage

### Combining Multiple Repos

After creating guides for 3-4 related repos:

```
Now create a meta-comparison document:
Compare FlashOCC, BEVFormer, BEVDepth, BEVStereo on:
1. BEV generation method
2. Temporal modeling approach
3. Performance vs efficiency tradeoff
4. Best use case for each

Use their Algorithm_Comparison.md files as reference.
```

### Tracking Evolution

```
Track evolution of BEV perception field:
1. List all methods chronologically (2020-2024)
2. Identify key innovations at each step
3. Show performance progression (mIoU over time)
4. Predict next research direction

Create: BEV_Perception_Evolution.md
```

## 🤝 Feedback Loop

As you use this template:

1. **Note what works well** → Keep in template
2. **Note what AI struggles with** → Add more guidance
3. **Note repetitive fixes you make** → Add to anti-hallucination checks
4. **Note domain gaps** → Add new domain-specific template

The template improves with each use!

## 📞 When to Ask for Help

If AI:
- Repeatedly hallucinates features not in code → Remind of verification protocol
- Skips numerical examples → Point to template requirement
- Creates generic content → Demand repo-specific details with file:line
- Doesn't self-correct → Explicitly trigger Round 2 accuracy sweep

---

## 🎓 Final Thoughts

This template encodes your **ideal learning experience**:
1. Complete understanding (inheritance → implementation)
2. Verified accuracy (every claim proven by code)
3. Practical mastery (can debug, modify, optimize)
4. Comparative knowledge (vs related methods)
5. Self-assessment (prove you've learned)

It transforms **"AI, explain this repo"** into **"AI, create a structured course that gets me to mastery in 6 hours"**.

Good luck with your research! 🚀
