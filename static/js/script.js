
document.getElementById('formA').style.display = 'block';

function showForm(formId) {
    var forms = document.querySelectorAll('.form-container');
    for (var i = 0; i < forms.length; i++) {
      if (forms[i].id === formId) {
        forms[i].style.display = 'block';
      } else {
        forms[i].style.display = 'none';
      }
    }
  
}
  