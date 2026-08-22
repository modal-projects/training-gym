export interface TutorialMetadata {
  order: number;
  title: string;
  deps: string[];
  content: string;
}

export function parseTutorialMetadata(
  source: string,
  tutorialPath: string,
): TutorialMetadata;
