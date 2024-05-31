class qresult:
    def __init__(self, n_findings, scores, files, base64):
        self.n = n_findings
        self.scores = scores
        self.files = files
        self.base64 = base64

    def __str__(self) -> str:
        return f"qresult(n_findings={self.n}, scores={self.scores}, files={self.files})"

    def __repr__(self) -> str:
        return (f"qresult(n_findings={self.n}, scores={self.scores}, files={self.files}, "
                f"base64 lengths={[len(b64) for b64 in self.base64]})")