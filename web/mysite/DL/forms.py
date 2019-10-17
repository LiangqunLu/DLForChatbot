from django import forms

class NameForm(forms.Form):
    txt_g = forms.CharField(initial='I went to the bank to play Frisbee.', label='input_sent', max_length=100, widget=forms.Textarea, required=False)


