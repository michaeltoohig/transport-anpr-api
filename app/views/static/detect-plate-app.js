var elem = new Vue({
  el: "#detect-plate-app",
  delimiters: ["[[", "]]"],
  data: {
    selectedFile: null,
    taskId: null,
    progress: 0,
    result: null,
  },
  methods: {
    resetData () {
      this.taskId = null
      this.progress = 0
      this.result = null
    },
    onFileSelected (event) {
      this.resetData()
      this.selectedFile = event.target.files[0]
    },
    async onUpload () {
      const fd = new FormData()
      fd.append('image', this.selectedFile)
      const requestOptions = {
        method: "POST",
        body: fd,
      };
      fetch(`${API_STR}/detect/plate`, requestOptions)
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
    async onVehicleImage (sentData) {
      console.log('GOT vehicle img', sentData)
      const requestOptions = {
        method: "POST",
      };
      fetch(`${API_STR}/detect/plate?taskId=${sentData.taskId}&file=${sentData.file}&makePrediction=${Boolean(sentData.makePrediction)}`, requestOptions)
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
      fetch(`${API_STR}/detect/plate/${this.taskId}`, requestOptions)
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
      if (!!this.taskId && this.progress === 1) {
        console.log('Done')
      } else {
        if (this.taskId != null) {
          console.log('Poll again')
          setTimeout(() => { this.poll() }, 1000)
        }
      }
    },
    predictPlate () {
      bus.$emit('send-plate-image', {taskId: this.taskId})
    },
  },
  mounted () {
    bus.$on('send-vehicle-image', data => {
      this.resetData()
      this.onVehicleImage(data)
    })
  }
});