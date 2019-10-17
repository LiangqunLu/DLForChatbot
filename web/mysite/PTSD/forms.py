from django import forms

class NameForm(forms.Form):

    jtype = forms.CharField(label='jtype')

    txt_g = forms.CharField(label='text', max_length=100, widget=forms.Textarea, required=False)

