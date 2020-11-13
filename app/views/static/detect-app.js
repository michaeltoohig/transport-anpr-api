var elem = new Vue({
  el: "#detect-app",
  delimiters: ["[[", "]]"],
  data: {
    selectedFile: null,
    taskId: null,
    progress: 0,
    result: null,
  },
  methods: {
    onFileSelected (event) {
      this.taskId = null
      this.progress = 0
      this.result = null
      this.selectedFile = event.target.files[0]
    },
    async onUpload () {
      const fd = new FormData()
      fd.append('image', this.selectedFile)
      const requestOptions = {
        method: "POST",
        body: fd,
      };
      fetch(`/api/detect`, requestOptions)
        .then(async response => {
          const data = await response.json()

          // check for error response
          if (!response.ok) {
            // get error message from body or default to response status
            console.log(data)
            const error = (data) || response.status
            return Promise.reject(error)
          }

          this.taskId = data.taskId
          this.poll()
        })
        .catch(error => {
          console.error('There was an error!', error)
        });
    },
    async poll () {
      const requestOptions = {
        method: "GET",
      }
      fetch(`/api/detect/${this.taskId}`, requestOptions)
        .then(async response => {
          const data = await response.json()

          // Check for error response
          if (!response.ok) {
            console.log(data)
            const error = (data) || response.status
            return Promise.reject(error)
          }

          console.log(data)
          this.progress = data.progress
          this.result = data
        })
        .catch(error => {
          console.error('There was an error!', error)
        })
      if (!!this.taskId && this.progress >= 0.9) {
        console.log('Done')
      } else {
        if (this.taskId != null) {
          console.log('Poll again')
          setTimeout(() => { this.poll() }, 1000)
        }
      }
    }
  }
});