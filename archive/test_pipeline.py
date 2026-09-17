import sys
sys.path.insert(0, 'C:\\Users\\hemal\\OneDrive\\Documents\\grammar_checker')
from new_pipeline_runner import NewGrammarRunner
runner = NewGrammarRunner()
result = runner.check('Yesterday my friend and me goes to a large stopping mall because we waited to buy some new clothe. When we reach the mall, there was many people writing outside because the stops was not spend yet.')
print(f'Issues: {len(result)}')
for r in result:
    print(f'  {r["original_text"]} -> {r["replacement"]} [{r["category"]}]')

print()
result2 = runner.check('Yesterday my friend and I went to the shopping mall. There were many people waiting outside because the shops were not open yet.')
print(f'Clean text issues: {len(result2)}')
for r in result2:
    print(f'  {r["original_text"]} -> {r["replacement"]} [{r["category"]}]')

print()
result3 = runner.check('The students were writing outside while they waited for the teacher.')
print(f'Writing text issues: {len(result3)}')
for r in result3:
    print(f'  {r["original_text"]} -> {r["replacement"]} [{r["category"]}]')
