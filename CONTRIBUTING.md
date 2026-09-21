# Contributing

Memory Daemon welcomes contributions.

Before adding features, ask:

**Does this preserve the project's intent?**

---

## Questions to Ask

Does it improve:

- Performance?
- Clarity?
- Modularity?
- Local-first behavior?
- Determinism?

If not, it probably belongs elsewhere.

---

## Guiding Principles

1. **Local-first** — No cloud dependencies
2. **Deterministic** — Same input → same output
3. **Modular** — Components are replaceable
4. **Inspectable** — Every decision is traceable
5. **Extensible** — Easy to add new features

---

## Coding Standards

### Small Modules
- Each file does one thing
- If a file exceeds 400 lines, consider splitting
- Private helpers go in `_` prefixed modules

### Clear Interfaces
- Use Pydantic models for data
- Type hints on all functions
- Docstrings for public methods
- No surprises

### Minimal Coupling
- Subsystems communicate through records
- No cross-layer imports (except through interfaces)
- Use dependency injection where possible

### Benchmark Changes
- Run `python benchmark/benchmark_runner.py`
- Verify accuracy doesn't regress
- Check latency stays within target (~150ms)
- Document any performance changes

### Document Architecture Changes
- Update relevant docs
- Explain why the change was made
- Note any breaking changes

---

## Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the benchmark suite
5. Update documentation
6. Submit a pull request

---

## Code Review Expectations

Reviewers will check for:

- [ ] Preserves design intent
- [ ] Passes benchmarks
- [ ] Includes tests
- [ ] Documentation updated
- [ ] No hidden dependencies
- [ ] Local-first preserved
- [ ] Deterministic behavior

---

## Getting Help

- Read the docs in `docs/`
- Check existing issues
- Open a discussion for design questions

---

## See Also

- `project_manifesto.md` — The why
- `05_Design_Principles.md` — The rules
- `07_design_intent.md` — Decision framework
