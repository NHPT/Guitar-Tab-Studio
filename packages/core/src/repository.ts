import type { ProjectRepository } from './contracts.js'
import type { StudioProject } from './domain.js'
import { migrateStudioProject } from './project.js'

export class MemoryProjectRepository implements ProjectRepository {
  private readonly projects = new Map<string, StudioProject>()

  constructor(seed: unknown[] = []) {
    for (const value of seed) {
      const project = migrateStudioProject(value)
      this.projects.set(project.id, project)
    }
  }

  getProject(id: string): StudioProject | undefined {
    return this.projects.get(id)
  }

  listProjects(): StudioProject[] {
    return [...this.projects.values()]
  }

  saveProject(project: StudioProject): void {
    const validated = migrateStudioProject(project)
    this.projects.set(validated.id, validated)
  }

  removeProject(id: string): void {
    this.projects.delete(id)
  }
}
