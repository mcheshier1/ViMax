import {useEffect, useRef, useState} from 'react';
import {AlertTriangle, RefreshCw, Save, User} from 'lucide-react';
import {readShotPlan, updateShotPlan} from './api';
import {characterChips, planDirty, slotName, toggleCharacter} from './timeline';
import type {PlanFrame, ShotPlan} from './types';

/**
 * What a shot's frames say and show, and the prompt each was drawn from — with the
 * characters in each frame editable.
 *
 * The render follows the plan: the frame description is what the image model is asked for,
 * and the visible characters decide whose portraits the reference selector is offered. Both
 * are the reason a shot comes back wrong, and re-running a shot without being able to say
 * what it should have been is how a shot gets redrawn five times with the same answer.
 */
export function ShotPlanPanel({sessionId, root, slot, stage, plan: given, onRedraw, onChanged}: {
  sessionId: string;
  root: string;
  slot: string;
  stage: string;
  /** The plan the view already has, so opening a row costs no request. */
  plan?: ShotPlan;
  onRedraw: (note: string) => void;
  onChanged?: () => void;
}) {
  const [plan, setPlan] = useState<ShotPlan | undefined>(given);
  const [draft, setDraft] = useState<{ffDesc: string; lfDesc: string; ffVis: number[]; lfVis: number[]}>({ffDesc: '', lfDesc: '', ffVis: [], lfVis: []});
  const [note, setNote] = useState('');

  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);

  // Seeded once per shot, not on every plan the view hands over: the view rebuilds that
  // object every few seconds as the render reports in, and re-seeding would throw away the
  // toggles and the description the user is halfway through typing.
  const seededFor = useRef('');
  useEffect(() => {
    if (seededFor.current === slot) return;
    seededFor.current = slot;
    const seed = (loaded: ShotPlan) => {
      // The panel may have moved to another shot while this was in flight.
      if (seededFor.current !== slot) return;
      setPlan(loaded);
      setDraft({ffDesc: loaded.firstFrame.description, lfDesc: loaded.lastFrame.description, ffVis: loaded.firstFrame.visible, lfVis: loaded.lastFrame.visible});
    };
    if (given) {
      seed(given);
      return;
    }
    // No plan handed over yet: fetch this one. Deliberately no cleanup cancelling it — this
    // effect re-runs when the view hands a plan over, and cancelling there would abandon the
    // only thing that was going to fill the panel.
    setError('');
    void readShotPlan(sessionId, root, slot)
      .then(seed)
      .catch((reason) => { if (seededFor.current === slot) setError(reason instanceof Error ? reason.message : String(reason)); });
  }, [sessionId, root, slot, given]);

  const dirty = Boolean(plan) && planDirty(plan as ShotPlan, draft);

  async function save() {
    if (!plan) return;
    setSaving(true);
    setError('');
    try {
      const saved = await updateShotPlan({sessionId, root, slot, ffDesc: draft.ffDesc, lfDesc: draft.lfDesc, ffVis: draft.ffVis, lfVis: draft.lfVis});
      setPlan(saved);
      setDraft({ffDesc: saved.firstFrame.description, lfDesc: saved.lastFrame.description, ffVis: saved.firstFrame.visible, lfVis: saved.lastFrame.visible});
      onChanged?.();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  if (error && !plan) return <p className="plan-error" role="alert">{error}</p>;
  if (!plan) return <p className="plan-loading">Loading what this shot says…</p>;

  return (
    <section className="shot-plan">
      {(['firstFrame', 'lastFrame'] as const).map((key) => (
        <FramePlan
          key={key}
          label={key === 'firstFrame' ? 'First frame' : 'Last frame'}
          frame={plan[key]}
          characters={plan.characters}
          description={key === 'firstFrame' ? draft.ffDesc : draft.lfDesc}
          visible={key === 'firstFrame' ? draft.ffVis : draft.lfVis}
          onDescription={(value) => setDraft((current) => key === 'firstFrame' ? {...current, ffDesc: value} : {...current, lfDesc: value})}
          onToggle={(idx, on) => setDraft((current) => key === 'firstFrame'
            ? {...current, ffVis: toggleCharacter(current.ffVis, idx, on)}
            : {...current, lfVis: toggleCharacter(current.lfVis, idx, on)})}
        />
      ))}
      {plan.brief && <p className="plan-brief"><span>Shot brief</span>{plan.brief}</p>}
      <div className="plan-actions">
        <button className="timeline-action" onClick={() => void save()} disabled={!dirty || saving}>
          <Save size={13} /> {saving ? 'Saving…' : dirty ? 'Save plan' : 'Saved'}
        </button>
        <button
          className="timeline-action is-primary"
          onClick={() => onRedraw(note)}
          title="Redraw this shot from the plan above, with your note"
        >
          <RefreshCw size={13} /> Redraw with this plan
        </button>
      </div>
      <input
        className="plan-note"
        value={note}
        onChange={(event) => setNote(event.target.value)}
        placeholder="What is wrong with this shot, in your words (optional)"
        aria-label={`Note for ${slotName(stage, slot)}`}
      />
      {error && <p className="plan-error" role="alert">{error}</p>}
    </section>
  );
}

function FramePlan({label, frame, characters, description, visible, onDescription, onToggle}: {
  label: string;
  frame: PlanFrame;
  characters: {idx: number; name: string}[];
  description: string;
  visible: number[];
  onDescription: (value: string) => void;
  onToggle: (idx: number, on: boolean) => void;
}) {
  return (
    <div className="plan-frame">
      <header>
        <strong>{label}</strong>
        <div className="plan-characters">
          {characterChips(characters, visible).map((character) => (
            <button
              key={character.idx}
              className={`plan-character${character.on ? ' is-on' : ''}`}
              aria-pressed={character.on}
              onClick={() => onToggle(character.idx, !character.on)}
              title={character.on ? `${character.name} is in this frame` : `${character.name} is not in this frame`}
            >
              <User size={11} /> {character.name}
            </button>
          ))}
          {characters.length === 0 && <span className="plan-hint"><AlertTriangle size={11} /> No characters found for this sequence</span>}
        </div>
      </header>
      <textarea
        value={description}
        rows={3}
        aria-label={`${label} description`}
        onChange={(event) => onDescription(event.target.value)}
      />
      {frame.prompt ? (
        <details className="plan-prompt">
          <summary>Prompt sent for this frame</summary>
          <pre>{frame.prompt}</pre>
        </details>
      ) : (
        // A frame taken from a camera still is drawn without a prompt at all, so "not drawn
        // yet" would be wrong as often as it was right.
        <p className="plan-hint">No prompt recorded for this frame.</p>
      )}
    </div>
  );
}
