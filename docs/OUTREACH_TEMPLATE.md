# Cover note template — READ THIS FIRST

Do not send this as-is. Everything in [brackets] must be filled in truthfully
by you, and the paragraphs marked (PERSONALIZE) need your actual words, not
mine. A generic or copy-pasted-feeling email to a researcher is worse than a
short honest one. This is a starting structure, not a finished product.

A few things to decide honestly before you send anything:

1. **How you describe how this was built.** If asked directly "did you build
   this yourself," the honest answer is that you used an AI coding assistant
   to scaffold it and you directed, reviewed, tested, and extended it. That
   is an increasingly normal way to build software in 2026, including in
   research groups. Claiming sole unassisted authorship, if asked and it
   isn't true, is the kind of thing that ends an RA relationship the moment
   it's discovered. You don't need to lead with "an AI wrote this," but you
   should not deny it if asked, and you should be able to explain and defend
   every design decision in the code as if you made it yourself — because at
   this point, having reviewed and tested it, you're accountable for it.
2. **Why this specific researcher.** Generic outreach ("I saw you work in
   sports science, I have a sports science project") gets ignored. Read 2-3
   of their actual papers first and reference something specific.
3. **What you're actually asking for.** "An RA position" is vague. Do you
   want to help with an existing project, propose extending this one under
   their supervision, or just get feedback and a foot in the door? Decide
   before writing.

---

## Template

Subject: [Your name] — pose-estimation project relevant to [specific paper/lab focus]

Dear Professor [Last name],

(PERSONALIZE — 1-2 sentences) I'm [year/program] at [institution], and I've
been following your work on [specific topic from their actual papers — e.g.
"ACL injury risk screening in field-sport athletes" or "markerless motion
capture for gait analysis"]. [One sentence on what specifically interested
you about a specific paper of theirs, not a generic compliment.]

I built an open-source tool that uses MediaPipe's monocular pose estimation
to screen sport-specific movement patterns (squat, sprint, golf swing,
football landing/cutting mechanics, and cricket bowling/batting positioning)
against thresholds motivated by the biomechanics literature — code, tests,
and the specific citations behind each threshold (including where my 2D
proxies diverge from what the cited studies actually measured) are here:

- Repository: [GitHub link]
- Live demo: [Streamlit Cloud link]
- Literature grounding: [link to docs/REFERENCES.md]

(PERSONALIZE) I'm aware this is a longer way from a validated research tool
than it might first look — [name one specific limitation you actually
understand, e.g. "I haven't validated any of these joint angles against a
marker-based reference, which I know is the obvious next step before any of
this could support a real claim"]. That gap is actually why I'm reaching
out: I'd welcome the chance to work on closing it under your supervision, or
to contribute to [specific ongoing project of theirs, if you know of one]
in whatever way would be useful to your lab.

I'd be glad to walk through the code or the design decisions in person if
that's useful. Thank you for your time either way.

Best regards,
[Your name]
[Contact info]

---

## Before you hit send

- [ ] I have actually read this entire codebase and can explain any file if asked
- [ ] I have run it on real video and know what it gets right and wrong
- [ ] I have personally tuned or changed at least one threshold and can say why
- [ ] I have read at least 2 of this specific researcher's actual papers
- [ ] I am not claiming sole unassisted authorship anywhere I wouldn't defend if asked directly
- [ ] The GitHub repo has real, multi-commit history, not one dump
- [ ] The live demo link actually works
