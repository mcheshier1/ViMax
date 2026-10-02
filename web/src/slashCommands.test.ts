import {describe, expect, it} from 'vitest';
import {matchingSlashCommands, shouldShowSlashCommands} from './slashCommands';

describe('slash command matching', () => {
  it('matches command prefixes and exposes highlighted segments', () => {
    expect(matchingSlashCommands('/co')[0]).toMatchObject({matchedPrefix: '/co', unmatchedSuffix: 'mpact'});
  });

  it('keeps queued slash commands available only when the bridge can queue safely', () => {
    expect(shouldShowSlashCommands('/', false)).toBe(true);
    expect(shouldShowSlashCommands('/co', true)).toBe(false);
    expect(shouldShowSlashCommands('/co', true, true)).toBe(true);
    expect(shouldShowSlashCommands('hello', false)).toBe(false);
  });
});
