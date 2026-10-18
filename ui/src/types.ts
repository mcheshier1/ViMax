export type WorkspaceLine =
  | {kind: 'user'; text: string}
  | {kind: 'assistant'; text: string; sampleId?: string; seq?: number; outcome?: 'final' | 'tool_preamble' | 'interrupted' | 'truncated'}
  | {kind: 'thinking'; text: string}
  | {kind: 'status'; text: string}
  | {kind: 'workflow'; text: string}
  | {kind: 'tool'; text: string; status?: 'running' | 'done' | 'error'}
  | {kind: 'terminal'; text: string}
  | {kind: 'session'; text: string}
  | {kind: 'error'; text: string};

export type ToolResult = {
  name?: string;
  ok?: boolean;
  content?: string;
  metadata?: Record<string, unknown>;
};

export type StreamEvent = {
  type?: string;
  turn_id?: string;
  sample_id?: string;
  final_sample_id?: string;
  seq?: number;
  text?: string;
  outcome?: 'final' | 'tool_preamble' | 'interrupted' | 'truncated';
  status?: string;
  delta?: string;
  phase?: string;
  message?: string;
  stream?: string;
  line?: string;
  assistant?: string;
  tool?: {name?: string; requested_name?: string; arguments?: Record<string, unknown>};
  progress?: {stage?: string; message?: string; metadata?: Record<string, unknown>};
  tool_result?: ToolResult;
  session?: {
    active_session_id?: string;
    session?: {
      session_id?: string;
      stage?: string;
      working_dir?: string;
    } | null;
  };
  prompt_trace?: {
    total_estimated_tokens?: number;
    totals?: {
      total_tokens?: number;
      total_estimated_tokens?: number;
    };
  };
};

export type MappingState = {
  assistantStreaming: boolean;
  sampleId?: string;
};
