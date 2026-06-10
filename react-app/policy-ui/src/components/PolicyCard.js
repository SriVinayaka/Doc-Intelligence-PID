

import React from 'react'

const PolicyCard = ({ data }) => {
  return (
    <div style={{
      border: '1px solid #ccc',
      padding: '10px',
      marginBottom: '10px',
      borderRadius: '8px'
    }}>
          <strong>{data.filename}</strong><br />
          Policy ID: {data.policy_id}<br />
          Type:      {data.policy_type}<br />
          DoB:       {data.date_of_birth}<br />
          Nominee:   {data.nominee}<br />
          Phone:     {data.phone}<br />
          Email:     {data.email}<br/>
          Score:     {data._score}<br />
    </div>
  )
}

export default PolicyCard

