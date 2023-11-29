import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent implements OnInit {
  consoleOutput: string = '';
  graphImage: string = '';
  isLoading = false;

  constructor(private http: HttpClient) { }

  ngOnInit(): void {
  }

  runModel(): void {
    this.isLoading = true;
    this.http.post<any>('http://localhost:5000/run-model', {}).subscribe(
      response => {
        // Fetch the text and image files
        this.fetchFileContent('http://localhost:5000/generated/' + response.console_output_file);
        this.graphImage = 'http://localhost:5000/generated/' + response.graph_file;
        this.isNotLoading();

      },
      error => {
        console.error('Error running the model:', error);
      }
    );
  }

  fetchFileContent(filePath: string): void {
    this.http.get(filePath, { responseType: 'text' }).subscribe(content => {
      this.consoleOutput = content;
    });
  }

  resetModel(): void {
    this.http.get<any>('http://localhost:5000/reset').subscribe(
      response => {
        console.log(response.message);
        window.location.reload();  // Refresh the page
      },
      error => {
        console.error('Error resetting the model:', error);
      }
    );
  }

  isNotLoading(): boolean {
    return this.isLoading = false;
}
}
