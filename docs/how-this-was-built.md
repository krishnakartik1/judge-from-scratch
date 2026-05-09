# How this was built

> A note on the AI-assisted workflow that produced `judge-from-scratch`. For the project itself and the build narrative, see the [README](../README.md) and the two-part story ([data](story-data.md) + [training](story-training.md)).

This project was built with Claude as a collaborator at every level. Two AI tools, each doing what it's good at:

- **Claude in chat (via a Claude Project).** The conceptual work — the [primer](fine-tuning-primer.md), the data design, the eval methodology — was all developed in long-form conversation. The Project's "knowledge" was the primer, the [build prompts](claude-code-prompts.md), and the [project status](project-status.md) — three files attached as project knowledge so any new chat could pick up context without re-explaining the project. When a stage finished, the status file got updated and the next chat session inherited that context automatically.
- **Claude Code for staged implementation.** Each pipeline stage was built from a single scoped prompt in [`docs/claude-code-prompts.md`](claude-code-prompts.md). One prompt per stage, one PR per stage. Inside each stage, Claude Code wrote the script, ran the dryrun, surfaced the dryrun output, waited for review, and then ran the full thing.

The split worked because the two tools have different strengths. Long chat is good at "should we hold out religion or religion+disability?" and "how do we close the verbosity-bias hole in the synthesis prompt?" Claude Code is good at "given this spec, write a 200-line resumable async script with proper error handling." Mixing them up — using long chat to write code, or using Claude Code to argue about methodology — wastes both.

## The workflow pattern

The same loop repeated for every stage:

1. **Brainstorm in chat.** Bring the open question (which generator pool? how should pairs be constructed? what are the bias buckets?) into a Claude chat with the primer + prompts + status as project knowledge. Iterate until the methodology is concrete.
2. **Codify decisions in project docs.** Update [`project-status.md`](project-status.md) with the resolved decision (with a number — those numbers become load-bearing references downstream). Update [`claude-code-prompts.md`](claude-code-prompts.md) with the scoped prompt for the next stage if the methodology changed how that stage should run.
3. **Hand the scoped prompt to Claude Code.** One stage, one prompt, in a fresh Claude Code session. The prompt always says "read these files first" — the primer, the status, the prompt manual — so Claude Code starts with the same context as the chat conversation that produced the prompt.
4. **Review the output.** Read the code yourself. Run the dryrun. Verify the artifact (JSONL file, model checkpoint, eval table) actually looks right. Don't blindly accept; the agent's plan can be plausible-looking and wrong.
5. **Update status and commit.** Mark the stage done in `project-status.md`, document anything that was learned (especially anything surprising), commit as a stage-shaped PR.
6. **Repeat.**

The status file is the connective tissue. A fresh chat or Claude Code session can pick up cleanly with just the four core context files (`README.md`, `fine-tuning-primer.md`, `claude-code-prompts.md`, `project-status.md`) — no conversation history needed.

## What stays yours, what to delegate

| You own | The assistant owns |
|---|---|
| Data design (which buckets, which categories, why) | Script implementation, async/retry plumbing |
| Eval methodology (κ, position bias, OOD definition) | API client glue, JSON parsing, file I/O |
| The labeling rubric | Boilerplate (chat-template wrapping, Modal images, configs) |
| Deciding when the AI's plan is wrong | Translating a clear plan into working code |
| What "good enough" looks like for the dataset | Resumability, retries, log formatting |

The skill in using a coding assistant well is partly knowing when *not* to delegate — pushing back when its plan is plausible-looking but wrong. The two-part story includes several examples of that, including the `lora_dropout = 0` decision (primer said 0.05; runtime warning + recent research + dryrun ablation all said 0; data won) and the truncation-bias eval finding (the agent's first eval-budget choice was systematically biased; the methodology lesson came from noticing the parse-failure pattern, not from the agent flagging it).
