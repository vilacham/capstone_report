class BestStreakClassifier:
    """ A best streak classifier
    
    Parameters
    ----------
    home_streak: {array-like}, shape = [n_samples]
    Number of streak wins for the home team
    
    away_streak: {array-like}, shape = [n_samples]
    Number of streak wins for the visitor team
    """
    def __init__(self, home_streak, away_streak):
        self.home_streak = home_streak
        self.away_streak = away_streak

    def predict(self):
        import numpy as np
        prediction = list()
        for i in range(len(self.home_streak)):
            if self.away_streak[i] > self.home_streak[i]:
                prediction.append(0)
            else:
                prediction.append(1)
        prediction = np.asarray(prediction)
        return prediction

    def score(self, y):
        from sklearn.metrics import accuracy_score
        score = accuracy_score(y, self.predict())
        return score
