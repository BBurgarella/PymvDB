class qresult:
    """
    A class to represent the result of a query for similar images.

    Attributes
    ----------
    n : int
        The number of findings that match the query.
    scores : list of float
        The similarity scores of the matching images.
    files : list of str
        The file paths of the matching images.
    base64 : list of str
        The base64 encoded strings of the matching images.
    metadata : list of dict
        The metadata associated with the matching images.

    Methods
    -------
    __str__()
        Returns a string representation of the query result.
    __repr__()
        Returns a detailed string representation of the query result for debugging.
    """
    def __init__(self, n_findings, scores, files, base64, metadata):
        """
        Parameters
        ----------
        n_findings : int
            The number of findings that match the query.
        scores : list of float
            The similarity scores of the matching images.
        files : list of str
            The file paths of the matching images.
        base64 : list of str
            The base64 encoded strings of the matching images.
        metadata : list of dict
            The metadata associated with the matching images.
        """
        self.n = n_findings
        self.scores = scores
        self.files = files
        self.base64 = base64
        self.metadata = metadata

    def __str__(self) -> str:
        """
        Returns a string representation of the query result.

        Returns
        -------
        str
            A string representation of the query result.
        """
        return (f"qresult(n_findings={self.n}, scores={self.scores}, "
                f"files={self.files}, metadata={self.metadata})")

    def __repr__(self) -> str:
        """
        Returns a detailed string representation of the query result for debugging.

        Returns
        -------
        str
            A detailed string representation of the query result for debugging.
        """
        return (f"qresult(n_findings={self.n}, scores={self.scores}, files={self.files}, "
                f"base64 lengths={[len(b64) for b64 in self.base64]}, metadata={self.metadata})")
