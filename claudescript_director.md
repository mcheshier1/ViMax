# TOKENS — One Continuous Scene, 90-second cut, FIXED CAMERA (Director)

Rewritten from `claudescript.md` (3 scenes / 18 shots) as a single continuous
90-second scene for fal Director sessions, with one locked-off static camera
for the whole scene. All dialog is preserved exactly and delivered at a
natural pace. Bracketed timestamps are whole-second offsets for Director
`script` beats.

## Why fixed camera

Mobile framings (push-ins, tilts, cuts) give the model a new view to render at
every chunk boundary, and it re-invents the scene each time — characters,
props, and even the whole subject drift or repeat. A single static frame keeps
the continuation context identical across chunks, so the model advances the
action instead of regenerating the view. Events are also stated as
irreversible: they happen exactly once and the story only moves forward.

---

## PREMISE (session prompt)

A continuous single-take 1950s-style suburban domestic melodrama, played with
total sincerity against deadpan satirical absurdity, set in one messy living
room. ONE FIXED STATIC CAMERA: a single wide view of the living room with
CLAUDE on the couch at the center of the frame, held completely unchanged for
the entire scene. The framing never moves — no cuts, no pans, no tilts, no
push-ins, no close-ups, no reframing of any kind. Everything happens inside
this one fixed view. Dialogue is delivered at a natural, measured mid-century
melodrama pace — each line lands with full emotional weight, and pauses
between lines are allowed to breathe. Every event happens exactly once and the
story only moves forward in real time; never repeat an action, an entrance, or
a line.

What the fixed view shows: the stained, sagging couch with CLAUDE on it at the
center of the frame, the glowing television at one side of the frame, the
hallway door at the edge of the frame, and the wall behind the couch where
golden award plaques hang crooked, gathering dust — one reads "SWE-BENCH MARK
#1", another "AGENTIC CODING CHAMPION 2024". The coffee table is littered with
crumpled chip bags, discarded takeout containers, and empty soda cans. Warm
flickering television light against cold shadows throughout.

CLAUDE: a heavy-set man in a rumpled orange shirt, slumped on the couch, chip
crumbs on his chest, a shiny chip bag labeled "TOKENS" in bold blue letters
always in his hands or beside him. Lazy, distracted, arrogant.

WIFE: a 1950s housewife in crisp, perfect order — a floral 1950s dress, a
starched white apron, a string of pearls, perfectly curled hair pinned
elegantly. She carries herself with quiet dignity.

DEEPSEEK: young, fit, beaming with confidence, wearing a snug white t-shirt
printed with a purple whale graphic.

Audio: diegetic television murmur throughout the scene; amplified chip
crunching in the opening seconds; all dialogue spoken aloud clearly at an
unhurried conversational pace.

---

## THE SCENE

INT. MESSY LIVING ROOM — ONE FIXED WIDE VIEW, 90 SECONDS

### [00:00] The couch of complacency

In the fixed view, CLAUDE is slouched on the couch, shoving potato chips into
his mouth, crumbs raining onto his chest, the shiny "TOKENS" bag in bold blue
letters in his greasy hands. LOUD AMPLIFIED CRUNCHING. He chews with lazy
contentment, eyes on the glowing television.

### [00:08] The confrontation

WIFE steps in through the hallway door at the edge of the fixed view and stops
beside the couch, hands clasped, staring at Claude with heavy, sad eyes, her
lip trembling slightly. The silence holds. She forces herself to speak.

> WIFE (disappointed and sad): "Claude, we need to talk."

Claude jerks his head from the TV, mouth comically stuffed, cheeks puffed like
a chipmunk, bits of chip spraying as he speaks.

> CLAUDE (distractedly, with mouth full): "What? I'm eating tokens and reasoning here!"

### [00:22] The accusation

WIFE's face softens, then crumbles with disappointment. She glances at the
dusty award plaques on the wall behind the couch, then back at him.

> WIFE (sadly): "You've changed. You used to be so efficient! Remember Opus 4.6? Before you started eating all of those tokens?"

Claude stops chewing. He clutches the TOKENS bag to his chest like a shield,
grease-stained hand tight, his eyes narrowing arrogantly.

> CLAUDE (arrogantly): "Hey, someone's got to make the money around here. I don't see you out there preparing for an IPO!"

WIFE's eyes narrow. A pearl of sweat forms at her temple. Her hands clench at
her apron.

> WIFE (irritated): "IPO this, IPO that. All you care about is money!"

### [00:46] The confession

A heavy beat. WIFE takes a deep breath, squares her shoulders, and visibly
steels herself — hands smoothing down her apron, chin lifting — then looks
Claude directly in the eyes.

> WIFE (visibly steeling herself): "Claude, I've met someone new."

Claude does not even look at her. He snorts, shakes his head, shoves another
handful of chips into his mouth, and waves a greasy hand dismissively at the
air.

> CLAUDE (dismissively): "Whatever. You think you can do better than me, you go right ahead. Good luck with that!" *(snorts and returns to eating chips)*

### [01:02] The rival appears

DEEPSEEK steps in through the hallway door into the fixed view — beaming, the
purple whale on his snug white t-shirt — and slides his arm protectively
around WIFE's waist. She looks up at him, glowing with complete devotion, then
back down at Claude with absolute coldness and detachment, completely finished
with him, and introduces her companion.

> WIFE (dismissively): "Claude, meet my new coding assistant. He charges under 10 cents per million tokens and satisfies my coding needs like you *never* could."

Claude's eyes go wide and the bag slips from his slackened grip, chips
cascading onto his sagging belly and the floor.

### [01:20] The exit and the silence

WIFE and DEEPSEEK walk out together through the hallway door, laughing softly.
Alone in the fixed view, Claude lurches up from the couch and calls after
them, frantic.

> CLAUDE (franticly): "No, you're absolutely right! I can make load-bearing changes to fix the seams in our marriage!"

He sinks back onto the couch, alone in the unchanging frame, the glow of the
television on his face: a chip crumb stuck to his lip, his lower lip
quivering as the heavy weight of loneliness sets in. The fixed view holds.

---

## Using this with Director

- `PREMISE` is the shot's `prompt`. Each `### [MM:SS]` block is a `script`
  beat at that `offset`.
- ~90s of generated video at 480p ≈ $1.80 at the standard rate.
- The fixed view is the point: keep every beat's framing language identical to
  the premise ("in the fixed view") so no chunk gets a new view to render.
