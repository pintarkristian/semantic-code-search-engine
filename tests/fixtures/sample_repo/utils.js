// Utility functions for the sample repo fixture

/**
 * Format a Date object as an ISO date string (YYYY-MM-DD).
 */
function formatDate(date) {
    return date.toISOString().split('T')[0];
}

/**
 * Return true if the given value is a non-empty string.
 */
function isNonEmptyString(value) {
    return typeof value === 'string' && value.length > 0;
}

class QueryBuilder {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
        this.params = {};
    }

    // Add a query parameter
    addParam(key, value) {
        this.params[key] = value;
        return this;
    }

    build() {
        const qs = Object.entries(this.params)
            .map(([k, v]) => `${k}=${encodeURIComponent(v)}`)
            .join('&');
        return qs ? `${this.baseUrl}?${qs}` : this.baseUrl;
    }
}
