//import parent-port cause we need to listen to master and send back messages
const { parentPort } = require('worker_threads')

const cache = new Map()

/*
    In all of the operations
    1. Check the operation type
    2. Destructure the object struct thats coming in to get the params
    3. Perform the operation on the map
    4. Return using postMessage()
*/

parentPort.on('message', async message => {
    if(message.type === 'get'){
        const { key } = message
        const value = cache.get(key)
        parentPort.postMessage({ value })
    }

    else if(message.type === 'set'){
        const { key, value } = message
        cache.set(key, value)
        parentPort.postMessage({})
    }

    else if(message.type === 'has'){
        const { key } = message
        const exists = cache.has(key)
        parentPort.postMessage({ exists })
    }
    else if(message.type === 'delete'){
        const { key } = message
        cache.delete(key)
        parentPort.postMessage({})
    }
})

