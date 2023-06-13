const { Worker } = require('worker_threads') //this allows us to setup the worker which will work independently from the main thread
const crypto = require('crypto') //this is for hashing the key

class DistributedCache{
    //users can specify number of worker threads setup warning tho
    constructor(workerCount){
        this.workers = [] //set up array of workers

        //initialize them
        for(let i=0; i<workerCount; i++){
            const worker = new Worker('./worker.js') //code to be executed to by each worker specified in worker.js
            this.workers.push(worker) 
        }
    }

    getWorkerForKey(key){
        const index = this.hashKey(key) % this.workers.length
        return this.workers[index]
    }

    /*
    For each of the `GET`, `SET`, `CONTAINS` and `DELETE` functions :-
        1. Get the worker
        2. Then send the operation to the worker along with the parameter (this is async so it is a Promise) using worker.postMessage()
        3. Once the event is done look if the return is error or not
    */


    //get function getting key param
    async get(key){
        const worker = this.getWorkerForKey(key)
        return new Promise((resolve, reject) => {
            worker.postMessage({
                type: 'get',
                key
            })

            worker.once('message', message => {
                if(message.error){
                    reject(new Error(message.error))
                } else {
                    resolve(message.value)
                }
            })
        })
        
    }

    //set function getting k-v pair param
    async set(key, value){
        const worker = this.getWorkerForKey(key)
        return new Promise((resolve, reject) => {
            worker.postMessage({
                type: 'set',
                key,
                value
            })

            worker.once('message', message => {
                if(message.error){
                    reject(new Error(message.error))
                } else {
                    resolve()
                }
            })
        })
        
    }

    //contains function getting k param
    async has(key){
        const worker = this.getWorkerForKey(key)
        return new Promise((resolve, reject) => {
            worker.postMessage({
                type: 'has',
                key
            })

            worker.once('message', message => {
                if(message.error){
                    reject(new Error(message.error))
                } else {
                    resolve(message.exists)
                }
            })
        })
    }

    //delete function getting key
    async delete(key){
        const worker = this.getWorkerForKey(key)
        return new Promise((resolve, reject) => {
            worker.postMessage({
                type: 'delete',
                key
            })

            worker.once('message', message => {
                if(message.error){
                    reject(new Error(message.error))
                } else {
                    resolve()
                }
            })
        })

    }

    //hash the key using sha256
    hashKey(key){
        const hash = crypto.createHash('sha256')
        hash.update(key)
        const hashedKey = hash.digest('hex')
        return parseInt(hashedKey, 16)
    }

    //shutdown all workers
    shutdown(){
        this.workers.forEach((worker) => {
            return worker.terminate()
        })
    }
}

module.exports = DistributedCache