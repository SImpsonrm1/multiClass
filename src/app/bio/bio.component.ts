import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-bio',
  templateUrl: './bio.component.html',
  styleUrls: ['./bio.component.css']
})
export class BioComponent implements OnInit {

  teamMembers = [
    {
      name: 'Ryan Simpson',
      photo: 'assets/headshot.jpg',
      description: 'Hi I\'m Ryan. I\'m a student software developer. at Appalachian state university.'
    },
    {
      name: 'Jordan Klein',
      photo: 'assets/headshot1.png',
      description: 'I\'m an undergraduate student studying mathematics, statistics, and data science.'

    },
    {
      name: 'Alex Somer',
      photo: 'assets/headshot2.jpg',
      description: ' I\'m an undergrad student studying computer science with a concentration in data science and machine learning.'
    }
  ];

  constructor() { }

  ngOnInit(): void {
  }

}
