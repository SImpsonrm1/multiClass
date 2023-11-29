import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup } from '@angular/forms';

@Component({
  selector: 'app-contact',
  templateUrl: './contact.component.html',
  styleUrls: ['./contact.component.css']
})
export class ContactComponent implements OnInit {
  contactForm!: FormGroup;
  savedForms: any[] = [];

  constructor(private fb: FormBuilder) { }

  ngOnInit(): void {
    this.contactForm = this.fb.group({
      name: [''],
      number: [''],
      email: [''],
      description: ['']
    });

    // Retrieve and display saved forms from local storage
    this.loadSavedForms();
  }

  onSubmit(): void {
    const formData = this.contactForm.value;

    // Convert form data to JSON string
    const formDataJSON = JSON.stringify(formData);

    // Generate a unique ID for this form submission
    const formId = new Date().getTime().toString();

    // Save to local storage
    localStorage.setItem(`contactForm_${formId}`, formDataJSON);

    // Update the savedForms array
    this.loadSavedForms();
  }

  loadSavedForms(): void {
    const keys = Object.keys(localStorage).filter(key => key.startsWith('contactForm_'));
    this.savedForms = keys.map(key => JSON.parse(localStorage.getItem(key) as string));
  }
}
