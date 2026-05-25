// TypeScript type definitions and helpers for the sample repo fixture

interface SearchResult {
    score: number;
    filePath: string;
    startLine: number;
    endLine: number;
    snippet: string;
}

class SearchResultFormatter {
    // Format a single result for terminal display
    format(result: SearchResult): string {
        return `[${result.score.toFixed(2)}] ${result.filePath}:${result.startLine}-${result.endLine}`;
    }

    formatAll(results: SearchResult[]): string[] {
        return results.map(r => this.format(r));
    }
}

// Normalise a raw query string before embedding
function normaliseQuery(query: string): string {
    return query.trim().toLowerCase().replace(/\s+/g, ' ');
}
