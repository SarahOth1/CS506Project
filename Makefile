PYTHON = python3

install:
$(PYTHON) -m pip install pandas numpy scikit-learn matplotlib

data:
$(PYTHON) dataset.py

train:
$(PYTHON) training.py

plots:
$(PYTHON) plottings.py

run: install data train plots

clean:
rm -f movies_cleaned.csv plotting_data.csv
rm -f actor_avg_rating.png actress_avg_rating.png director_avg_rating.png writer_avg_rating.png