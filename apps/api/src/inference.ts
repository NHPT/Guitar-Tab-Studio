import type {
  InferenceProvider,
  InferenceRequest,
  StudioProject,
} from '@gts/core'
import {
  getCapabilities,
  runPipeline,
  type PipelineExecutionOptions,
} from './pipeline.js'

function inferenceTimeoutMs(): number {
  const configured = Number(process.env.GTS_INFERENCE_TIMEOUT_SECONDS ?? 7200)
  const seconds =
    Number.isFinite(configured) && configured >= 60 && configured <= 86_400
      ? configured
      : 7200
  return Math.round(seconds * 1000)
}

export class PipelineInferenceProvider implements InferenceProvider<string> {
  constructor(
    private readonly executionResolver?: (
      jobId: string,
    ) => PipelineExecutionOptions,
  ) {}

  capabilities() {
    return getCapabilities()
  }

  analyze(
    request: InferenceRequest<string>,
    onProgress: Parameters<InferenceProvider<string>['analyze']>[1],
  ): Promise<StudioProject> {
    return runPipeline(
      request.job,
      (job) => onProgress({ job }),
      request.media,
      {
        ...this.executionResolver?.(request.job.id),
        signal: request.signal,
        timeoutMs: inferenceTimeoutMs(),
      },
    )
  }
}
