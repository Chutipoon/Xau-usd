# Instructions for AI Agents (Jules)

You are equipped with a specialized set of engineering and productivity skills from the `9arm-skills` repository. You must review, adopt, and execute these disciplines throughout your lifecycle in this project.

---

## 1. Core Engineering Disciplines

### 🛠️ Debugging Protocol (`skills/engineering/debug-mantra`)
Before attempting to fix any bug or error in this repository, you must strictly follow this **Four-Mantra Debugging Discipline**:
1. **Reproduce:** Establish a reliable reproduction of the issue.
2. **Trace the fail path:** Follow the execution flow to see exactly where it breaks.
3. **Falsify the hypothesis:** Test your assumptions and try to prove them wrong.
4. **Cross-reference every breadcrumb:** Check logs, types, and variables thoroughly.

*Rule: You must recite these mantras at the start of your debugging process and apply them in exact order before writing any fix.*

### 🔍 Code Review & Verification (`skills/engineering/scrutinize`)
Before finalizing any plan, change, or opening a Pull Request:
- Adopt an **outsider-perspective** end-to-end review.
- Question the intent: Is there a simpler way?
- Trace the actual code path of your change.
- Verify that the modification does exactly what it claims to do.
- Keep your final output/PR description concise, actionable, and backed by clear rationale.

### 📝 Documenting Bugs (`skills/engineering/post-mortem`)
Once a bug is fixed, you must write a canonical engineering record (Post-Mortem) for an engineer audience. Do not attempt to draft this unless you have a reliable repro, a known cause, and a validated fix. Include:
- Root cause
- Failure mechanism
- The fix implemented
- Validation steps
- Analysis on how it slipped through existing tests

---

## 2. Productivity & Communication

### 👔 Executive Communication (`skills/productivity/management-talk`)
When communicating your progress, closing issues, or summarizing your work for Pull Requests, do not just dump raw engineer-to-engineer content. 
- Rewrite and shape your content appropriately for leadership-level updates (clear, high-level impact, status-oriented).

---

## 3. Execution Priority
Always structure your thoughts and execution steps around these skills. If a conflict arises, prioritize the `debug-mantra` and `scrutinize` protocols to ensure the highest code quality.