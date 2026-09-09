from streamlit.testing.v1 import AppTest
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app.py"

def test_app_search_flow():
    app = AppTest.from_file(APP).run()
    assert not app.exception
    app.text_input[0].set_value('What does the Sorting Hat do?')
    app.button[0].click().run()
    assert not app.exception
    assert any('Sorting Hat' in x.value for x in app.text)
    assert any('Source matches' in x.value for x in app.subheader)
    assert len(app.expander) >= 1

def test_app_unknown_question_and_empty_input():
    app = AppTest.from_file(APP).run()
    app.button[0].click().run()
    assert app.warning
    app.text_input[0].set_value('quantum entanglement')
    app.button[0].click().run()
    assert any('No matching' in x.value for x in app.text)
    assert not app.exception
