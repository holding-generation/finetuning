from alignscore import AlignScore
import nltk

nltk.download('punkt')

scorer = AlignScore(model='roberta-base', batch_size=32, device='cuda:0', ckpt_path='./AlignScore-base.ckpt', evaluation_mode='nli_sp')
score = scorer.score(contexts=['hello world.'], claims=['hello world.'])

print(f"The score is: {score}")
