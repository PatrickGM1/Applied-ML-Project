import { BrowserRouter, Routes, Route } from "react-router-dom"
import Home from "./Home"
import Predict from "./Predict"

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/demo" element={<Predict />} />
      </Routes>
    </BrowserRouter>
  )
}
