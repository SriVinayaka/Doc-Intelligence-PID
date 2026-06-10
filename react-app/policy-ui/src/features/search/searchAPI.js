import axios from 'axios'

const API_URL = 'http://localhost:8000/Search'

// export const fetchSearchResults = async (payload) => {
//   const response = await axios.post(API_URL, payload)
//   return response.data
// }

export const fetchSearchResults = async (payload) => {
  try {
    const response = await axios.post(
      'http://localhost:8000/query',
      payload,
      {
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    return response.data
  } catch (error) {
    console.error("API ERROR:", error)
    throw error
  }
}
