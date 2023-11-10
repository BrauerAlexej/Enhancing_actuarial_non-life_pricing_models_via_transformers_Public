import sys

def costume_progress_bar(text,current,total, bar_length):
    """
    creates a progress bar for the fitting of the model

    Args:
        text (str): the text that should be displayed in front of the progress bar
        current (int): the current epoch
        total (int): the total number of epochs
        bar_length (int): the length of the progress bar
        
    Returns:
        None
    """
    progress = current/total
    block = int(round(bar_length*progress))
    # do a padding for text 
    text = text.ljust(70)
    text = "\r" + text + ": [{0}] {1:.1f}%".format( "#"*block + "-"*(bar_length-block), progress*100)
    sys.stdout.write(text)
    sys.stdout.flush()
    

class Easy_ProgressTracker():
    """
    A class to track the progress of the model fitting
    """
    def __init__(self, 
                 current_epoch = 0, 
                 current_score = 9999,
                 best_epoch = 0, 
                 best_score = 9999,
                 patience=10,
                 maximize=False,
                 progress=False):
        """
        Args:
            epochs (int): the number of epochs
            batch_size (int): the batch size
            bar_length (int): the length of the progress bar
        """
        self.best_epoch = best_epoch
        self.current_epoch = current_epoch
        self.best_score = best_score if maximize == False else -best_score
        self.current_score = current_score
        self.patience = patience
        self.progress = progress
        self.patience_over = False
        self.maximize = maximize
    
    def __call__(self, current_epoch, current_score):
        """
        updates the progress tracker and checks if the current score is better than the best score
        if so, it updates the best score and the best epoch
        if the current score is worse than the best score and current_epoch-best_epoch>patience then progress is set it returns False 
        
        Args:
            current_epoch (int): the current epoch
            current_score (float): the current score
        
        Returns:
            None
        """
        self.current_epoch = current_epoch
        self.current_score = current_score
        if ((self.current_score < self.best_score) & (self.maximize == False)) or ((self.current_score > self.best_score) & (self.maximize == True)):
            self.best_score = self.current_score
            self.best_epoch = self.current_epoch
            self.progress = True
        else: 
            self.progress = False
            if self.current_epoch - self.best_epoch > self.patience:
                self.patience_over = True

